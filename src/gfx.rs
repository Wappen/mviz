//! GPU rendering (wgpu): a batched 2D shape renderer, text (glyphon), and a
//! phosphor afterglow for the sound field.
//!
//! All drawing uses logical window coordinates (pixels / scale factor),
//! origin top-left. Colours are display (sRGB-encoded) values blended as-is,
//! which is how every colour in this app was tuned.

use bytemuck::{Pod, Zeroable};
use glyphon::{Attrs, Buffer, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport};
use std::sync::Arc;
use winit::window::Window;

pub type Rgba = [f32; 4];

pub static FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSans-Regular.ttf");
const FONT_FAMILY: &str = "Noto Sans";

/// Phosphor afterglow. Per pixel, at every analysis step (a few per frame,
/// see `PhosphorFrame::steps`):
///   glow  = blur(glow) · e^(−dt/fade) + max(previous live − live, 0)
/// and per frame
///   shown = live (averaged over the frame's steps) + glow
/// The live light (the dots) is shown as it is; whatever light leaves a
/// pixel (a sound stopping, getting quieter or moving away) stays in the
/// texture and fades there, fast then slowly, spreading a little. So moving
/// sounds leave trails, a stopped sound hands over to its glow without a
/// step, a steady sound does not pile up, and all light adds like light.
/// Only flicker feeds the glow while a sound holds; the fade time
/// (`params::Params::fade`) is the lever if that gets too bright. The blur
/// (`fade_blur` per 1/60 s, see `blur_offset`) lets fading light spread out.
/// Taking light per step rather than per frame matters for moving light: the
/// glow takes what a pixel lost, which adds up to the brightest it got, and
/// a spot averaged over a frame's travel is dimmer where two frames meet, so
/// fast trails would show a ridge per frame.
/// The phosphor textures are this fraction of the window size (it is all soft glow).
const PHOSPHOR_SCALE: f32 = 0.5;
/// Most halvings of the picture when blurring.
const BLUR_LEVELS: usize = 7;
const PHOSPHOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: Rgba,
    kind: u32,
}

const SOLID: u32 = 0;
const GLOW: u32 = 1;
/// A fade from clear to its colour along `uv.x`, eased (smoothstep).
const FADE: u32 = 2;
const CORNER_LIGHT: u32 = 3;

/// A list of shapes to draw in one go.
#[derive(Default)]
pub struct Batch {
    verts: Vec<Vertex>,
}

impl Batch {
    fn quad(&mut self, p: [[f32; 2]; 4], uv: [[f32; 2]; 4], c: [Rgba; 4], kind: u32) {
        for i in [0, 1, 2, 0, 2, 3] {
            self.verts.push(Vertex { pos: p[i], uv: uv[i], color: c[i], kind });
        }
    }

    /// Glows in this batch as (centre x, centre y, radius x, radius y, colour) (tests).
    #[cfg(test)]
    pub fn glows(&self) -> Vec<(f32, f32, f32, f32, Rgba)> {
        self.verts
            .chunks(6)
            .filter(|q| q[0].kind == GLOW)
            .map(|q| {
                let (a, b) = (q[0].pos, q[2].pos);
                ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (b[0] - a[0]) * 0.5, (b[1] - a[1]) * 0.5, q[0].color)
            })
            .collect()
    }

    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, c: Rgba) {
        self.gradient_h(x, y, w, h, c, c);
    }

    /// Rectangle whose colour runs from `left` to `right`.
    pub fn gradient_h(&mut self, x: f32, y: f32, w: f32, h: f32, left: Rgba, right: Rgba) {
        let p = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
        self.quad(p, [[0.0; 2]; 4], [left, right, right, left], SOLID);
    }

    /// From clear at the top to `c` at the bottom, eased at both ends (no
    /// visible edge where it starts or where it meets solid colour).
    pub fn fade_v(&mut self, x: f32, y: f32, w: f32, h: f32, c: Rgba) {
        let p = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
        self.quad(p, [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], [c; 4], FADE);
    }

    /// From clear at the left to `c` at the right, eased like `fade_v`.
    pub fn fade_h(&mut self, x: f32, y: f32, w: f32, h: f32, c: Rgba) {
        let p = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
        self.quad(p, [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], [c; 4], FADE);
    }

    /// The start screen's background over the whole `w` × `h` area, at
    /// strength `a`: light pooling from two opposite corners over deep navy
    /// (see `corner_light` in the shader). It moves with `Scene::time`.
    pub fn corner_light(&mut self, w: f32, h: f32, a: f32) {
        let p = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        self.quad(p, uv, [[1.0, 1.0, 1.0, a]; 4], CORNER_LIGHT);
    }

    /// Soft round glow (gaussian-like falloff) with radii `rx`, `ry`.
    pub fn glow(&mut self, cx: f32, cy: f32, rx: f32, ry: f32, c: Rgba) {
        self.ellipse(cx, cy, rx, ry, c, GLOW);
    }

    fn ellipse(&mut self, cx: f32, cy: f32, rx: f32, ry: f32, c: Rgba, kind: u32) {
        let p = [[cx - rx, cy - ry], [cx + rx, cy - ry], [cx + rx, cy + ry], [cx - rx, cy + ry]];
        let uv = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        self.quad(p, uv, [c; 4], kind);
    }
}

pub struct Text {
    pub text: String,
    /// Top-left corner, logical pixels.
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub color: Rgba,
}

/// The phosphor part of a frame (stereo modes).
pub struct PhosphorFrame {
    /// Lit throughout the frame (light that changes too slowly to differ
    /// between its steps): drawn once.
    pub live: Batch,
    /// What else is lit at each of the frame's analyses, evenly spaced over
    /// it (none: only `live`). The phosphor takes each in turn, so light
    /// that leaves a pixel goes into the glow when it leaves, not at the
    /// frame's end; shown is their average. Only small, fast light belongs
    /// here: every step draws it again.
    pub steps: Vec<Batch>,
    /// Seconds to advance the phosphor; `None` = paused (show it unchanged).
    pub advance: Option<f32>,
}

/// The offset (phosphor pixels) of the glow's spreading cross of bilinear
/// reads for one step of `dt` seconds, spreading as much per second as one
/// read `per_frame` out does per 1/60 s. A read at offset o puts its weight
/// on the two pixels around o, so its spread (second moment) is
/// m(o) = (1 − f)·i² + f·(i + 1)² with i = ⌊o⌋, f = o − i; spreads add up
/// over steps.
pub fn blur_offset(per_frame: f32, dt: f32) -> f32 {
    let m = |o: f32| {
        let (i, f) = (o.floor(), o.fract());
        (1.0 - f) * i * i + f * (i + 1.0) * (i + 1.0)
    };
    let t = m(per_frame.max(0.0)) * dt * 60.0;
    let i = t.sqrt().floor();
    i + (t - i * i) / (2.0 * i + 1.0)
}

/// Everything one frame shows, back to front.
#[derive(Default)]
pub struct Scene {
    pub clear: Rgba,
    /// Opaque layer (background).
    pub background: Batch,
    pub phosphor: Option<PhosphorFrame>,
    /// How much the background and phosphor are blurred (0..1): behind the
    /// start screen. The controls and text stay sharp.
    pub blur: f32,
    /// Seconds, for shapes that move by themselves (`Batch::corner_light`).
    pub time: f64,
    /// Blended on top (controls).
    pub ui: Batch,
    pub texts: Vec<Text>,
}

struct Phosphor {
    size: (u32, u32),
    /// This step's and the last step's dots (`PhosphorFrame::steps`), used in turn.
    live: [wgpu::TextureView; 2],
    /// This frame's and the last frame's light lit throughout (`PhosphorFrame::live`).
    slow: [wgpu::TextureView; 2],
    glow: [wgpu::TextureView; 2],
    /// The live light shown: the average over the last frame's steps.
    shown: wgpu::TextureView,
    /// Which `live` / `slow` / `glow` hold the latest state.
    cur_live: usize,
    cur_slow: usize,
    cur_glow: usize,
    /// update_bg[g][l][a][b]: reads glow g, previous live 1-l, new live l,
    /// previous slow a, new slow b.
    update_bg: [[[[wgpu::BindGroup; 2]; 2]; 2]; 2],
    /// composite_bg[g]: shows glow g + `shown`.
    composite_bg: [wgpu::BindGroup; 2],
}

/// Blurring: the picture drawn into `scene` (full size), then halved into
/// `levels` and grown back (dual Kawase), each step spreading it further.
struct Blur {
    size: (u32, u32),
    /// The picture, then each level's (half the size of the one before).
    views: Vec<wgpu::TextureView>,
    /// Reads view i (all three textures the same).
    read: Vec<wgpu::BindGroup>,
    /// The final pass: sharp picture and blurred (level 1).
    present: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurParams {
    amount: f32,
    pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FxParams {
    keep: f32,
    /// Soft roll-off level for bright light.
    knee: f32,
    texel: [f32; 2],
}

/// Where frames go: a window, or a texture that can be read back (video export).
enum Output {
    Window { window: Arc<Window>, instance: wgpu::Instance, surface: wgpu::Surface<'static> },
    Offscreen { texture: wgpu::Texture },
}

pub struct Gfx {
    output: Output,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    globals: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    blend_pipe: wgpu::RenderPipeline,
    add_hdr_pipe: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    vcap: u64,

    fx_layout: wgpu::BindGroupLayout,
    update_layout: wgpu::BindGroupLayout,
    fx_params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    update_pipe: wgpu::RenderPipeline,
    composite_pipe: wgpu::RenderPipeline,
    /// Copies (and rescales) phosphor textures when the window is resized.
    copy_pipe: wgpu::RenderPipeline,
    phosphor: Option<Phosphor>,

    blend_hdr_pipe: wgpu::RenderPipeline,
    composite_hdr_pipe: wgpu::RenderPipeline,
    /// The first halving, which also turns the picture into linear light.
    down_first_pipe: wgpu::RenderPipeline,
    down_pipe: wgpu::RenderPipeline,
    up_pipe: wgpu::RenderPipeline,
    present_pipe: wgpu::RenderPipeline,
    blur_params: wgpu::Buffer,
    blur: Option<Blur>,

    font_system: FontSystem,
    swash: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    text_renderer: TextRenderer,
}

impl Gfx {
    /// Renders into `window`.
    pub fn new(window: Arc<Window>, display: winit::event_loop::OwnedDisplayHandle) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display)));
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("no GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).expect("no GPU device");

        let caps = surface.get_capabilities(&adapter);
        // Non-sRGB output: colours are written as tuned, without conversion.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);
        Self::build(Output::Window { window, instance, surface }, device, queue, config)
    }

    /// Renders into an offscreen texture of `width` × `height` pixels, read
    /// back with `read_frame` (video export).
    pub fn new_offscreen(width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("no GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).expect("no GPU device");
        // Same as on screen: non-sRGB, colours written as tuned.
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        Self::build(Output::Offscreen { texture }, device, queue, config)
    }

    fn build(output: Output, device: wgpu::Device, queue: wgpu::Queue, config: wgpu::SurfaceConfiguration) -> Self {
        let format = config.format;

        // --- shapes ---
        let shapes = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shapes"),
            source: wgpu::ShaderSource::Wgsl(SHAPES_WGSL.into()),
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() }],
        });
        let shapes_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[Some(&globals_layout)],
            ..Default::default()
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Uint32],
        };
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
        };
        let shape_pipe = |target: wgpu::TextureFormat, blend: wgpu::BlendState, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&shapes_layout),
                vertex: wgpu::VertexState {
                    module: &shapes,
                    entry_point: Some("vs"),
                    buffers: &[Some(vertex_layout.clone())],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shapes,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState { format: target, blend: Some(blend), write_mask: wgpu::ColorWrites::ALL })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                cache: None,
                multiview_mask: None,
            })
        };
        let blend_pipe = shape_pipe(format, wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING, "blend");
        let add_hdr_pipe = shape_pipe(PHOSPHOR_FORMAT, additive, "add hdr");
        let blend_hdr_pipe = shape_pipe(PHOSPHOR_FORMAT, wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING, "blend hdr");
        let vcap = 64 * 1024;
        let vbuf = Self::vertex_buffer(&device, vcap);

        // --- phosphor ---
        let fx = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("phosphor"),
            source: wgpu::ShaderSource::Wgsl(PHOSPHOR_WGSL.into()),
        });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
            count: None,
        };
        let fx_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("phosphor"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                tex_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let sampler_entry = wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let params_entry = wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let update_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("phosphor update"),
            entries: &[tex_entry(0), tex_entry(1), tex_entry(2), sampler_entry, params_entry, tex_entry(5), tex_entry(6)],
        });
        let fx_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phosphor params"),
            size: std::mem::size_of::<FxParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let fx_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[Some(&fx_layout)],
            ..Default::default()
        });
        let fx_pipe_in = |fx: &wgpu::ShaderModule, entry: &str, target: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&fx_pipe_layout),
                vertex: wgpu::VertexState { module: fx, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: fx,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState { format: target, blend, write_mask: wgpu::ColorWrites::ALL })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                cache: None,
                multiview_mask: None,
            })
        };
        let fx_pipe = |entry: &str, target: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| fx_pipe_in(&fx, entry, target, blend);
        // The update writes the new glow, and adds its step's share (the
        // blend constant) of the live light to what is shown.
        let share = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Constant, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            alpha: additive.alpha,
        };
        let update_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("update"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[Some(&update_layout)],
                ..Default::default()
            })),
            vertex: wgpu::VertexState { module: &fx, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &fx,
                entry_point: Some("update"),
                targets: &[
                    Some(wgpu::ColorTargetState { format: PHOSPHOR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: PHOSPHOR_FORMAT, blend: Some(share), write_mask: wgpu::ColorWrites::ALL }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });
        let composite_pipe = fx_pipe("composite", format, Some(additive));
        let copy_pipe = fx_pipe("copy", PHOSPHOR_FORMAT, None);
        let composite_hdr_pipe = fx_pipe("composite", PHOSPHOR_FORMAT, Some(additive));

        // --- blur ---
        let blur_fx = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur"),
            source: wgpu::ShaderSource::Wgsl(BLUR_WGSL.into()),
        });
        let down_first_pipe = fx_pipe_in(&blur_fx, "down_first", PHOSPHOR_FORMAT, None);
        let down_pipe = fx_pipe_in(&blur_fx, "down", PHOSPHOR_FORMAT, None);
        let up_pipe = fx_pipe_in(&blur_fx, "up", PHOSPHOR_FORMAT, None);
        let present_pipe = fx_pipe_in(&blur_fx, "present", format, None);
        let blur_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur params"),
            size: std::mem::size_of::<BlurParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- text ---
        let font_system = FontSystem::new_with_fonts([glyphon::fontdb::Source::Binary(Arc::new(FONT_BYTES))]);
        let cache = glyphon::Cache::new(&device);
        let viewport = Viewport::new(&device, &cache);
        let mut atlas = TextAtlas::with_color_mode(&device, &queue, &cache, format, glyphon::ColorMode::Web);
        let text_renderer = TextRenderer::new(&mut atlas, &device, wgpu::MultisampleState::default(), None);

        Self {
            output,
            device,
            queue,
            config,
            globals,
            globals_bg,
            blend_pipe,
            add_hdr_pipe,
            vbuf,
            vcap,
            fx_layout,
            update_layout,
            fx_params,
            sampler,
            update_pipe,
            composite_pipe,
            copy_pipe,
            phosphor: None,
            blend_hdr_pipe,
            composite_hdr_pipe,
            down_first_pipe,
            down_pipe,
            up_pipe,
            present_pipe,
            blur_params,
            blur: None,
            font_system,
            swash: SwashCache::new(),
            viewport,
            atlas,
            text_renderer,
        }
    }

    fn vertex_buffer(device: &wgpu::Device, verts: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertices"),
            size: verts * std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// The window (windowed mode only).
    pub fn window(&self) -> &Arc<Window> {
        match &self.output {
            Output::Window { window, .. } => window,
            Output::Offscreen { .. } => panic!("offscreen rendering has no window"),
        }
    }

    pub fn scale(&self) -> f32 {
        match &self.output {
            Output::Window { window, .. } => window.scale_factor() as f32,
            Output::Offscreen { .. } => 1.0,
        }
    }

    /// Window size in logical pixels.
    pub fn size(&self) -> (f32, f32) {
        let s = self.scale();
        (self.config.width as f32 / s, self.config.height as f32 / s)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        if let Output::Window { surface, .. } = &self.output {
            surface.configure(&self.device, &self.config);
        }
    }

    /// The last rendered frame as RGBA bytes, row by row (offscreen only).
    /// Waits until the GPU has finished everything submitted (timing).
    #[cfg(test)]
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    pub fn read_frame(&self) -> Vec<u8> {
        let Output::Offscreen { texture } = &self.output else { return Vec::new() };
        let (w, h) = (self.config.width, self.config.height);
        let row = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let Ok(data) = slice.get_mapped_range() else { return Vec::new() };
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h as usize {
            out.extend_from_slice(&data[y * row as usize..y * row as usize + (w * 4) as usize]);
        }
        out
    }

    /// Width of `text` at `size` (logical pixels).
    pub fn text_width(&mut self, text: &str, size: f32) -> f32 {
        let buffer = self.shape(text, size);
        buffer.layout_runs().map(|r| r.line_w).fold(0.0, f32::max) / self.scale()
    }

    fn shape(&mut self, text: &str, size: f32) -> Buffer {
        let s = self.scale();
        let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(size * s, size * 1.3 * s));
        buffer.set_size(None, None);
        buffer.set_text(text, &Attrs::new().family(Family::Name(FONT_FAMILY)), Shaping::Advanced, None);
        buffer.shape_until_scroll(&mut self.font_system, false);
        buffer
    }

    fn ensure_phosphor(&mut self) {
        let size = (
            ((self.config.width as f32 * PHOSPHOR_SCALE) as u32).max(1),
            ((self.config.height as f32 * PHOSPHOR_SCALE) as u32).max(1),
        );
        if self.phosphor.as_ref().is_some_and(|p| p.size == size) {
            return;
        }
        let texture = |label| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: PHOSPHOR_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let live = [texture("live a"), texture("live b")];
        let slow = [texture("slow a"), texture("slow b")];
        let glow = [texture("glow a"), texture("glow b")];
        let shown = texture("shown");
        let bind = |a: &wgpu::TextureView, b: &wgpu::TextureView, c: &wgpu::TextureView| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("phosphor"),
                layout: &self.fx_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(a) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(b) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(c) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 4, resource: self.fx_params.as_entire_binding() },
                ],
            })
        };
        let update_bg = [0, 1].map(|g| {
            [0, 1].map(|l| {
                [0, 1].map(|a| {
                    [0, 1].map(|b| {
                        let tex = |binding, view| wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(view) };
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("phosphor update"),
                            layout: &self.update_layout,
                            entries: &[
                                tex(0, &glow[g]),
                                tex(1, &live[1 - l]),
                                tex(2, &live[l]),
                                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                                wgpu::BindGroupEntry { binding: 4, resource: self.fx_params.as_entire_binding() },
                                tex(5, &slow[a]),
                                tex(6, &slow[b]),
                            ],
                        })
                    })
                })
            })
        });
        let composite_bg = [0, 1].map(|g| bind(&glow[g], &shown, &shown));
        // New textures start black...
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for view in live.iter().chain(&slow).chain(&glow).chain([&shown]) {
            clear_pass(&mut encoder, view, wgpu::Color::BLACK);
        }
        // ...or, on a resize, with the old picture scaled to the new size, so
        // nothing is lost (while paused nothing new is drawn into them).
        if let Some(old) = self.phosphor.take() {
            let pairs = [(&old.live[old.cur_live], &live[0]), (&old.slow[old.cur_slow], &slow[0]), (&old.glow[old.cur_glow], &glow[0]), (&old.shown, &shown)];
            for (src, dst) in pairs {
                let bg = bind(src, src, src);
                let mut pass = begin(&mut encoder, dst, None);
                pass.set_pipeline(&self.copy_pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        self.phosphor = Some(Phosphor { size, live, slow, glow, shown, cur_live: 0, cur_slow: 0, cur_glow: 0, update_bg, composite_bg });
    }

    fn ensure_blur(&mut self) {
        let size = (self.config.width, self.config.height);
        if self.blur.as_ref().is_some_and(|b| b.size == size) {
            return;
        }
        // Halve until the smallest level is about 16 px high (a blur about
        // as wide as a sixth of the window), at most 7 times, and at least
        // once: the first halving is what turns the picture into light.
        let mut dims = vec![size];
        while dims.len() < 2 || (dims.len() <= BLUR_LEVELS && dims.last().unwrap().1 > 32) {
            let (w, h) = *dims.last().unwrap();
            dims.push(((w / 2).max(1), (h / 2).max(1)));
        }
        let views: Vec<wgpu::TextureView> = dims
            .iter()
            .map(|&(w, h)| {
                self.device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("blur"),
                        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: PHOSPHOR_FORMAT,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            })
            .collect();
        let bind = |a: &wgpu::TextureView, b: &wgpu::TextureView| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blur"),
                layout: &self.fx_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(a) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(b) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(b) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 4, resource: self.blur_params.as_entire_binding() },
                ],
            })
        };
        let read = views.iter().map(|v| bind(v, v)).collect();
        let present = bind(&views[0], &views[1]);
        self.blur = Some(Blur { size, views, read, present });
    }

    pub fn render(&mut self, scene: Scene) {
        let frame = match &mut self.output {
            Output::Offscreen { .. } => None,
            Output::Window { window, instance, surface } => match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(frame) => Some(frame),
                wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                    surface.configure(&self.device, &self.config);
                    return;
                }
                wgpu::CurrentSurfaceTexture::Lost => {
                    *surface = instance.create_surface(window.clone()).expect("recreate surface");
                    surface.configure(&self.device, &self.config);
                    return;
                }
                wgpu::CurrentSurfaceTexture::Validation => panic!("surface validation error"),
            },
        };
        let view = match (&frame, &self.output) {
            (Some(f), _) => f.texture.create_view(&Default::default()),
            (None, Output::Offscreen { texture }) => texture.create_view(&Default::default()),
            (None, Output::Window { .. }) => return,
        };
        let (w, h) = self.size();
        self.queue.write_buffer(&self.globals, 0, bytemuck::cast_slice(&[w, h, (scene.time % 10_000.0) as f32, 0.0]));

        // All shapes go into one vertex buffer; remember each layer's range.
        // Layers: background, ui, then the phosphor's light lit throughout
        // and at each step.
        let mut layers = vec![&scene.background, &scene.ui];
        if let Some(p) = &scene.phosphor {
            layers.push(&p.live);
            layers.extend(&p.steps);
        }
        let total: usize = layers.iter().map(|b| b.verts.len()).sum();
        if total as u64 > self.vcap {
            self.vcap = (total as u64).next_power_of_two();
            self.vbuf = Self::vertex_buffer(&self.device, self.vcap);
        }
        let mut ranges = vec![0..0; layers.len()];
        let mut offset = 0u32;
        for (i, layer) in layers.iter().enumerate() {
            let n = layer.verts.len() as u32;
            if n > 0 {
                self.queue.write_buffer(&self.vbuf, offset as u64 * std::mem::size_of::<Vertex>() as u64, bytemuck::cast_slice(&layer.verts));
            }
            ranges[i] = offset..offset + n;
            offset += n;
        }

        // Text.
        let s = self.scale();
        let buffers: Vec<(Buffer, &Text)> = scene.texts.iter().map(|t| (self.shape(&t.text, t.size), t)).collect();
        let areas = buffers.iter().map(|(b, t)| TextArea {
            buffer: b,
            left: t.x * s,
            top: t.y * s,
            scale: 1.0,
            bounds: TextBounds { left: 0, top: 0, right: self.config.width as i32, bottom: self.config.height as i32 },
            default_color: glyphon::Color::rgba(
                (t.color[0] * 255.0) as u8,
                (t.color[1] * 255.0) as u8,
                (t.color[2] * 255.0) as u8,
                (t.color[3] * 255.0) as u8,
            ),
            custom_glyphs: &[],
        });
        self.viewport.update(&self.queue, Resolution { width: self.config.width, height: self.config.height });
        let _ = self.text_renderer.prepare(&self.device, &self.queue, &mut self.font_system, &mut self.atlas, &self.viewport, areas, &mut self.swash);

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Phosphor: this frame's light lit throughout (`slow`), then for each
        // step its dots (`live`), and
        // glow = blur(glow)·keep + max(previous slow + live − slow − live, 0),
        // while adding the step's share of slow + live to what is shown. The
        // slow light only changes at a frame's first step: in later ones the
        // update reads the new slow light as both before and after.
        let mut composite = None;
        if let Some(pf) = &scene.phosphor {
            self.ensure_phosphor();
            let p = self.phosphor.as_mut().unwrap();
            let always = ranges[2].clone();
            // Without steps: one, with only the slow light.
            let steps: Vec<_> = if pf.steps.is_empty() { std::iter::once(0..0).collect() } else { ranges[3..].to_vec() };
            let n = steps.len();
            // Written every frame, so the knee also changes while paused.
            let cfg = crate::params::get();
            let dt = pf.advance.unwrap_or(0.0) / n as f32;
            let o = blur_offset(cfg.fade_blur, dt);
            let params = FxParams {
                keep: (-dt / cfg.fade.max(1e-3)).exp(),
                knee: cfg.knee.clamp(0.05, 0.99),
                texel: [o / p.size.0 as f32, o / p.size.1 as f32],
            };
            self.queue.write_buffer(&self.fx_params, 0, bytemuck::bytes_of(&params));
            if pf.advance.is_some() {
                let slow = 1 - p.cur_slow;
                {
                    let mut pass = begin(&mut encoder, &p.slow[slow], Some(wgpu::Color::BLACK));
                    draw_range(&mut pass, &self.add_hdr_pipe, &self.globals_bg, &self.vbuf, always);
                }
                let share = 1.0 / n as f64;
                for (i, step) in steps.into_iter().enumerate() {
                    let live = 1 - p.cur_live;
                    {
                        let mut pass = begin(&mut encoder, &p.live[live], Some(wgpu::Color::BLACK));
                        draw_range(&mut pass, &self.add_hdr_pipe, &self.globals_bg, &self.vbuf, step);
                    }
                    let glow = 1 - p.cur_glow;
                    {
                        let target = |view, load| {
                            Some(wgpu::RenderPassColorAttachment {
                                view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                            })
                        };
                        let shown_load = if i == 0 { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load };
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("phosphor update"),
                            color_attachments: &[target(&p.glow[glow], wgpu::LoadOp::Clear(wgpu::Color::BLACK)), target(&p.shown, shown_load)],
                            ..Default::default()
                        });
                        pass.set_pipeline(&self.update_pipe);
                        pass.set_blend_constant(wgpu::Color { r: share, g: share, b: share, a: share });
                        let before = if i == 0 { p.cur_slow } else { slow };
                        pass.set_bind_group(0, &p.update_bg[p.cur_glow][live][before][slow], &[]);
                        pass.draw(0..3, 0..1);
                    }
                    p.cur_live = live;
                    p.cur_glow = glow;
                }
                p.cur_slow = slow;
            }
            composite = Some(p.cur_glow);
        }

        let c = scene.clear;
        let clear = wgpu::Color { r: c[0] as f64, g: c[1] as f64, b: c[2] as f64, a: 1.0 };
        let blurred = scene.blur > 0.002;
        if blurred {
            // The picture into a texture, halved step by step and grown back.
            self.ensure_blur();
            self.queue.write_buffer(&self.blur_params, 0, bytemuck::bytes_of(&BlurParams { amount: scene.blur.min(1.0), pad: [0.0; 3] }));
            let b = self.blur.as_ref().unwrap();
            {
                let mut pass = begin(&mut encoder, &b.views[0], Some(clear));
                draw_range(&mut pass, &self.blend_hdr_pipe, &self.globals_bg, &self.vbuf, ranges[0].clone());
                if let (Some(g), Some(p)) = (composite, &self.phosphor) {
                    pass.set_pipeline(&self.composite_hdr_pipe);
                    pass.set_bind_group(0, &p.composite_bg[g], &[]);
                    pass.draw(0..3, 0..1);
                }
            }
            let n = b.views.len();
            for i in 1..n {
                let mut pass = begin(&mut encoder, &b.views[i], None);
                pass.set_pipeline(if i == 1 { &self.down_first_pipe } else { &self.down_pipe });
                pass.set_bind_group(0, &b.read[i - 1], &[]);
                pass.draw(0..3, 0..1);
            }
            for i in (1..n - 1).rev() {
                let mut pass = begin(&mut encoder, &b.views[i], None);
                pass.set_pipeline(&self.up_pipe);
                pass.set_bind_group(0, &b.read[i + 1], &[]);
                pass.draw(0..3, 0..1);
            }
        }
        {
            let mut pass = begin(&mut encoder, &view, Some(clear));
            if let (true, Some(b)) = (blurred, &self.blur) {
                pass.set_pipeline(&self.present_pipe);
                pass.set_bind_group(0, &b.present, &[]);
                pass.draw(0..3, 0..1);
            } else {
                draw_range(&mut pass, &self.blend_pipe, &self.globals_bg, &self.vbuf, ranges[0].clone());
                if let (Some(g), Some(p)) = (composite, &self.phosphor) {
                    pass.set_pipeline(&self.composite_pipe);
                    pass.set_bind_group(0, &p.composite_bg[g], &[]);
                    pass.draw(0..3, 0..1);
                }
            }
            draw_range(&mut pass, &self.blend_pipe, &self.globals_bg, &self.vbuf, ranges[1].clone());
            let _ = self.text_renderer.render(&self.atlas, &self.viewport, &mut pass);
        }

        self.queue.submit(Some(encoder.finish()));
        if let (Some(frame), Output::Window { window, .. }) = (frame, &self.output) {
            window.pre_present_notify();
            self.queue.present(frame);
        }
        self.atlas.trim();
    }
}

fn begin<'a>(encoder: &'a mut wgpu::CommandEncoder, view: &'a wgpu::TextureView, clear: Option<wgpu::Color>) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: clear.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn clear_pass(encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, color: wgpu::Color) {
    let _ = begin(encoder, view, Some(color));
}

fn draw_range(pass: &mut wgpu::RenderPass, pipe: &wgpu::RenderPipeline, globals: &wgpu::BindGroup, vbuf: &wgpu::Buffer, range: std::ops::Range<u32>) {
    if range.is_empty() {
        return;
    }
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, globals, &[]);
    pass.set_vertex_buffer(0, vbuf.slice(..));
    pass.draw(range, 0..1);
}

const SHAPES_WGSL: &str = r#"
struct Globals { size: vec2<f32>, time: f32, pad: f32 };
@group(0) @binding(0) var<uniform> g: Globals;

struct VIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) kind: u32,
};
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) kind: u32,
};

@vertex
fn vs(v: VIn) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(v.pos.x / g.size.x * 2.0 - 1.0, 1.0 - v.pos.y / g.size.y * 2.0, 0.0, 1.0);
    o.uv = v.uv;
    o.color = v.color;
    o.kind = v.kind;
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    var a = 1.0;
    if (i.kind == 1u) {
        // Soft glow: gaussian-like, reaching exactly 0 at the rim.
        let r2 = dot(i.uv, i.uv);
        a = select(0.0, exp(-r2 * 4.5) * (1.0 - r2), r2 < 1.0);
    } else if (i.kind == 2u) {
        // Eased fade: flat at both ends, so neither end shows as an edge.
        let t = clamp(i.uv.x, 0.0, 1.0);
        a = t * t * (3.0 - 2.0 * t);
    } else if (i.kind == 3u) {
        return vec4<f32>(corner_light(i.uv) * i.color.a, 1.0);
    }
    let alpha = i.color.a * a;
    return vec4<f32>(i.color.rgb * alpha, alpha);
}

fn hash(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    let r = q + dot(q, q + 45.32);
    return fract(r.x * r.y);
}

// Smooth value noise, 0..1.
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2<f32>(1.0, 0.0)), u.x),
               mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// Two octaves only: broad, soft variation with no fine grain.
fn soft(p0: vec2<f32>) -> f32 {
    return 0.65 * noise(p0) + 0.35 * noise(mat2x2<f32>(1.6, 1.2, -1.2, 1.6) * p0 + vec2<f32>(3.1, 7.7));
}

// Light pooling from two opposite corners over deep navy, like a dark
// gradient wallpaper: blue from the top-left, magenta from the
// bottom-right. Each pool's centre wanders around its corner and its rim
// is bent by broad noise, so it is never round; both slowly shift hue
// (blue to indigo, magenta to rust). Nearly even from edge to middle, a
// touch stronger at the edges.
fn corner_light(uv: vec2<f32>) -> vec3<f32> {
    let t = g.time;
    let aspect = g.size.x / max(g.size.y, 1.0);
    // Measured in a unit from the window's area (its height at 16:10), so
    // the pools keep their share of very wide or tall windows.
    let k = sqrt(aspect) / 1.265;
    let size = vec2<f32>(aspect, 1.0) / k;
    let p = uv * size;
    let c1 = vec2<f32>(0.1 * size.x + 0.15 * sin(t * 0.05), 0.05 * size.y + 0.12 * cos(t * 0.04));
    let c2 = vec2<f32>(0.9 * size.x + 0.15 * cos(t * 0.045), 0.95 * size.y + 0.12 * sin(t * 0.055));
    let bend = 0.35 * (soft(p * 0.8 + vec2<f32>(0.02 * t, 0.0)) - 0.5);
    let l1 = 1.0 - smoothstep(0.0, 1.1, length(p - c1) + bend);
    let l2 = 1.0 - smoothstep(0.0, 0.9, length(p - c2) - bend);
    let h = 0.5 + 0.5 * sin(t * 0.03);
    let navy = vec3<f32>(0.010, 0.020, 0.070);
    let c = navy
        + mix(vec3<f32>(0.02, 0.17, 0.30), vec3<f32>(0.10, 0.08, 0.32), h) * l1 * l1
        + mix(vec3<f32>(0.26, 0.04, 0.14), vec3<f32>(0.24, 0.08, 0.04), h) * l2 * l2;
    let d = min(min(uv.x, 1.0 - uv.x) * aspect, min(uv.y, 1.0 - uv.y));
    return c * mix(0.9, 1.0, 1.0 - smoothstep(0.0, 0.5, d));
}
"#;

/// WGSL shared by the full-screen passes: smooth upscaling and dither.
macro_rules! filter_wgsl {
    () => {
        r#"
// A texture read at `uv` through a cubic B-spline instead of bilinearly:
// smooth (no kinks every texel, which the eye sees as faint ridges) and
// never negative (no dark rims around bright light, unlike bicubic
// interpolation). Four bilinear reads, each placed to weigh its 2×2 texels
// by the spline. `smp` must filter linearly and clamp to the edge.
fn smooth_sample(t: texture_2d<f32>, uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(t));
    let q = uv * size - 0.5;
    let f = fract(q);
    let base = floor(q) + 0.5;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let a = (base - 1.0 + w1 / g0) / size;
    let b = (base + 1.0 + w3 / g1) / size;
    return g0.y * (g0.x * textureSample(t, smp, a).rgb + g1.x * textureSample(t, smp, vec2<f32>(b.x, a.y)).rgb)
         + g1.y * (g0.x * textureSample(t, smp, vec2<f32>(a.x, b.y)).rgb + g1.x * textureSample(t, smp, b).rgb);
}

// Noise of up to one 8-bit step either way (triangular, so it adds no
// visible grain of its own), for the last pass before an 8-bit screen:
// smooth dark gradients then show no bands.
fn dither(pos: vec2<f32>) -> f32 {
    let n1 = fract(52.9829189 * fract(dot(pos, vec2<f32>(0.06711056, 0.00583715))));
    let n2 = fract(52.9829189 * fract(dot(pos + vec2<f32>(37.0, 17.0), vec2<f32>(0.06711056, 0.00583715))));
    return (n1 + n2 - 1.0) / 255.0;
}
"#
    };
}

const PHOSPHOR_WGSL: &str = concat!(
    filter_wgsl!(),
    r#"
struct Params { keep: f32, knee: f32, texel: vec2<f32> };
@group(0) @binding(0) var glow: texture_2d<f32>;
// update: previous dots; composite: current dots
@group(0) @binding(1) var live: texture_2d<f32>;
// update: current dots
@group(0) @binding(2) var live_now: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;
@group(0) @binding(4) var<uniform> p: Params;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    // One triangle covering the whole target.
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VOut;
    o.pos = vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(q.x, 1.0 - q.y);
    return o;
}

// slow: what is lit throughout the frame, before and now (see `render`)
@group(0) @binding(5) var slow_before: texture_2d<f32>;
@group(0) @binding(6) var slow_now: texture_2d<f32>;

struct Update { @location(0) glow: vec4<f32>, @location(1) shown: vec4<f32> };

// glow' = blur(glow) · keep + light that left this pixel; and the light now
// lit, for what is shown (scaled by the step's share in the blending).
@fragment
fn update(i: VOut) -> Update {
    let o = p.texel;
    let blurred = textureSample(glow, smp, i.uv).rgb * 0.5
        + (textureSample(glow, smp, i.uv + vec2<f32>(o.x, 0.0)).rgb
         + textureSample(glow, smp, i.uv - vec2<f32>(o.x, 0.0)).rgb
         + textureSample(glow, smp, i.uv + vec2<f32>(0.0, o.y)).rgb
         + textureSample(glow, smp, i.uv - vec2<f32>(0.0, o.y)).rgb) * 0.125;
    let before = textureSample(live, smp, i.uv).rgb + textureSample(slow_before, smp, i.uv).rgb;
    let now = textureSample(live_now, smp, i.uv).rgb + textureSample(slow_now, smp, i.uv).rgb;
    let lost = max(before - now, vec3<f32>(0.0));
    return Update(vec4<f32>(blurred * p.keep + lost, 1.0), vec4<f32>(now, 1.0));
}

// A texture copied (and rescaled) as it is.
@fragment
fn copy(i: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(glow, smp, i.uv).rgb, 1.0);
}

// shown = live + glow, rolled off softly above 0.8 so bright overlaps
// compress instead of clipping. The textures are smaller than the window:
// scaled up smoothly, and dithered for the 8-bit screen.
@fragment
fn composite(i: VOut) -> @location(0) vec4<f32> {
    let c = smooth_sample(glow, i.uv) + smooth_sample(live, i.uv);
    let knee = p.knee;
    let over = max(c - vec3<f32>(knee), vec3<f32>(0.0));
    let soft = vec3<f32>(knee) + (1.0 - knee) * (vec3<f32>(1.0) - exp(-over / (1.0 - knee)));
    return vec4<f32>(select(c, soft, c > vec3<f32>(knee)) + dither(i.pos.xy), 1.0);
}
"#
);

const BLUR_WGSL: &str = concat!(
    filter_wgsl!(),
    r#"
struct Params { amount: f32, pad: f32, pad2: vec2<f32> };
// down / up: the level read; present: the sharp picture
@group(0) @binding(0) var src: texture_2d<f32>;
// present: the blurred picture
@group(0) @binding(1) var blurred: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;
@group(0) @binding(4) var<uniform> p: Params;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VOut;
    o.pos = vec4<f32>(q * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(q.x, 1.0 - q.y);
    return o;
}

// The picture is drawn with sRGB-encoded values, but frosted glass averages
// light: blur in linear light, so bright detail spreads into a glow instead
// of being dimmed by the dark around it.
fn to_linear(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(pow((x + 0.055) / 1.055, vec3<f32>(2.4)), x / 12.92, x <= vec3<f32>(0.04045));
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}

// Dual Kawase: half the size, each pixel from five taps of the level above.
fn down_taps(uv: vec2<f32>, linearize: bool) -> vec3<f32> {
    let o = 1.0 / vec2<f32>(textureDimensions(src));
    var t = array<vec3<f32>, 5>(
        textureSample(src, smp, uv).rgb,
        textureSample(src, smp, uv + vec2<f32>(-o.x, -o.y)).rgb,
        textureSample(src, smp, uv + vec2<f32>(o.x, -o.y)).rgb,
        textureSample(src, smp, uv + vec2<f32>(-o.x, o.y)).rgb,
        textureSample(src, smp, uv + vec2<f32>(o.x, o.y)).rgb,
    );
    if linearize {
        for (var k = 0; k < 5; k++) {
            t[k] = to_linear(t[k]);
        }
    }
    return (t[0] * 4.0 + t[1] + t[2] + t[3] + t[4]) / 8.0;
}

// The first halving reads the encoded picture.
@fragment
fn down_first(i: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(down_taps(i.uv, true), 1.0);
}

@fragment
fn down(i: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(down_taps(i.uv, false), 1.0);
}

// ...and back up to twice the size, from eight taps around each pixel.
@fragment
fn up(i: VOut) -> @location(0) vec4<f32> {
    let o = 1.0 / vec2<f32>(textureDimensions(src));
    var c = textureSample(src, smp, i.uv + vec2<f32>(-2.0 * o.x, 0.0)).rgb;
    c += textureSample(src, smp, i.uv + vec2<f32>(2.0 * o.x, 0.0)).rgb;
    c += textureSample(src, smp, i.uv + vec2<f32>(0.0, -2.0 * o.y)).rgb;
    c += textureSample(src, smp, i.uv + vec2<f32>(0.0, 2.0 * o.y)).rgb;
    c += textureSample(src, smp, i.uv + vec2<f32>(-o.x, -o.y)).rgb * 2.0;
    c += textureSample(src, smp, i.uv + vec2<f32>(o.x, -o.y)).rgb * 2.0;
    c += textureSample(src, smp, i.uv + vec2<f32>(-o.x, o.y)).rgb * 2.0;
    c += textureSample(src, smp, i.uv + vec2<f32>(o.x, o.y)).rgb * 2.0;
    return vec4<f32>(c / 12.0, 1.0);
}

// The sharp picture blended into the blurred one (half the size, scaled
// up smoothly) by `amount`, dithered for the 8-bit screen.
@fragment
fn present(i: VOut) -> @location(0) vec4<f32> {
    let c = mix(textureSample(src, smp, i.uv).rgb, to_srgb(smooth_sample(blurred, i.uv)), p.amount);
    return vec4<f32>(c + dither(i.pos.xy), 1.0);
}
"#
);

#[cfg(test)]
mod tests {
    use super::blur_offset;

    #[test]
    fn glow_spreads_as_much_per_second_in_any_steps() {
        // One step of a 60th of a second is the offset as given.
        for o in [0.0, 0.5, 1.0, 1.2, 2.7] {
            assert!((blur_offset(o, 1.0 / 60.0) - o).abs() < 1e-5, "{o}");
        }
        // Four steps spread as one 1/60 s step would (second moments add).
        let m = |o: f32| {
            let (i, f) = (o.floor(), o.fract());
            (1.0 - f) * i * i + f * (i + 1.0) * (i + 1.0)
        };
        let quarter = blur_offset(1.2, 1.0 / 240.0);
        assert!((4.0 * m(quarter) - m(1.2)).abs() < 1e-5, "{quarter}");
        assert!(blur_offset(1.2, 1.0 / 30.0) > 1.2);
    }
}
