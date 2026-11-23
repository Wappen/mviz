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

/// Phosphor afterglow. Per pixel, every frame:
///   glow  = blur(glow) · e^(−dt/fade) + max(previous live − live, 0)
///   shown = live + glow
/// The live light (the dots) is shown as it is; whatever light leaves a
/// pixel (a sound stopping, getting quieter or moving away) stays in the
/// texture and fades there, fast then slowly, spreading a little. So moving
/// sounds leave trails, a stopped sound hands over to its glow without a
/// step, a steady sound does not pile up, and all light adds like light.
/// Only flicker feeds the glow while a sound holds; the fade time
/// (`params::Params::fade`) is the lever if that gets too bright. The blur
/// per frame (`fade_blur`) lets fading light spread out.
/// The phosphor textures are this fraction of the window size (it is all soft glow).
const PHOSPHOR_SCALE: f32 = 0.5;
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
    /// What is lit right now.
    pub live: Batch,
    /// Seconds to advance the phosphor; `None` = paused (show it unchanged).
    pub advance: Option<f32>,
}

/// Everything one frame shows, back to front.
#[derive(Default)]
pub struct Scene {
    pub clear: Rgba,
    /// Opaque layer (background).
    pub background: Batch,
    pub phosphor: Option<PhosphorFrame>,
    /// Blended on top (controls).
    pub ui: Batch,
    pub texts: Vec<Text>,
}

struct Phosphor {
    size: (u32, u32),
    /// This frame's and last frame's dots, used in turn.
    live: [wgpu::TextureView; 2],
    glow: [wgpu::TextureView; 2],
    /// Which `live` / `glow` hold the latest state.
    cur_live: usize,
    cur_glow: usize,
    /// update_bg[g][l]: reads glow g, previous live 1-l, new live l.
    update_bg: [[wgpu::BindGroup; 2]; 2],
    /// composite_bg[g][l]: shows glow g + live l.
    composite_bg: [[wgpu::BindGroup; 2]; 2],
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
    fx_params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    update_pipe: wgpu::RenderPipeline,
    composite_pipe: wgpu::RenderPipeline,
    /// Copies (and rescales) phosphor textures when the window is resized.
    copy_pipe: wgpu::RenderPipeline,
    phosphor: Option<Phosphor>,

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
                visibility: wgpu::ShaderStages::VERTEX,
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
        let fx_pipe = |entry: &str, target: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&fx_pipe_layout),
                vertex: wgpu::VertexState { module: &fx, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &fx,
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
        let update_pipe = fx_pipe("update", PHOSPHOR_FORMAT, None);
        let composite_pipe = fx_pipe("composite", format, Some(additive));
        let copy_pipe = fx_pipe("copy", PHOSPHOR_FORMAT, None);

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
            fx_params,
            sampler,
            update_pipe,
            composite_pipe,
            copy_pipe,
            phosphor: None,
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
        let glow = [texture("glow a"), texture("glow b")];
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
        let update_bg = [0, 1].map(|g| [0, 1].map(|l| bind(&glow[g], &live[1 - l], &live[l])));
        let composite_bg = [0, 1].map(|g| [0, 1].map(|l| bind(&glow[g], &live[l], &live[l])));
        // New textures start black...
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for view in live.iter().chain(&glow) {
            clear_pass(&mut encoder, view, wgpu::Color::BLACK);
        }
        // ...or, on a resize, with the old picture scaled to the new size, so
        // nothing is lost (while paused nothing new is drawn into them).
        if let Some(old) = self.phosphor.take() {
            for (src, dst) in [(&old.live[old.cur_live], &live[0]), (&old.glow[old.cur_glow], &glow[0])] {
                let bg = bind(src, src, src);
                let mut pass = begin(&mut encoder, dst, None);
                pass.set_pipeline(&self.copy_pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        self.phosphor = Some(Phosphor { size, live, glow, cur_live: 0, cur_glow: 0, update_bg, composite_bg });
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
        self.queue.write_buffer(&self.globals, 0, bytemuck::cast_slice(&[w, h, 0.0, 0.0]));

        // All shapes go into one vertex buffer; remember each layer's range.
        let empty = Batch::default();
        let live = scene.phosphor.as_ref().map_or(&empty, |p| &p.live);
        let layers = [&scene.background, live, &scene.ui];
        let total: usize = layers.iter().map(|b| b.verts.len()).sum();
        if total as u64 > self.vcap {
            self.vcap = (total as u64).next_power_of_two();
            self.vbuf = Self::vertex_buffer(&self.device, self.vcap);
        }
        let mut ranges: [std::ops::Range<u32>; 3] = std::array::from_fn(|_| 0..0);
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

        // Phosphor: draw this frame's dots, then
        // glow = blur(glow)·keep + max(previous dots − dots, 0).
        let mut composite = None;
        if let Some(pf) = &scene.phosphor {
            self.ensure_phosphor();
            let p = self.phosphor.as_mut().unwrap();
            // Written every frame, so the knee also changes while paused.
            let cfg = crate::params::get();
            let params = FxParams {
                keep: pf.advance.map_or(1.0, |dt| (-dt / cfg.fade.max(1e-3)).exp()),
                knee: cfg.knee.clamp(0.05, 0.99),
                texel: [cfg.fade_blur / p.size.0 as f32, cfg.fade_blur / p.size.1 as f32],
            };
            self.queue.write_buffer(&self.fx_params, 0, bytemuck::bytes_of(&params));
            if pf.advance.is_some() {
                let live = 1 - p.cur_live;
                {
                    let mut pass = begin(&mut encoder, &p.live[live], Some(wgpu::Color::BLACK));
                    draw_range(&mut pass, &self.add_hdr_pipe, &self.globals_bg, &self.vbuf, ranges[1].clone());
                }
                let glow = 1 - p.cur_glow;
                {
                    let mut pass = begin(&mut encoder, &p.glow[glow], None);
                    pass.set_pipeline(&self.update_pipe);
                    pass.set_bind_group(0, &p.update_bg[p.cur_glow][live], &[]);
                    pass.draw(0..3, 0..1);
                }
                p.cur_live = live;
                p.cur_glow = glow;
            }
            composite = Some((p.cur_glow, p.cur_live));
        }

        {
            let c = scene.clear;
            let mut pass = begin(&mut encoder, &view, Some(wgpu::Color { r: c[0] as f64, g: c[1] as f64, b: c[2] as f64, a: 1.0 }));
            draw_range(&mut pass, &self.blend_pipe, &self.globals_bg, &self.vbuf, ranges[0].clone());
            if let (Some((g, l)), Some(p)) = (composite, &self.phosphor) {
                pass.set_pipeline(&self.composite_pipe);
                pass.set_bind_group(0, &p.composite_bg[g][l], &[]);
                pass.draw(0..3, 0..1);
            }
            draw_range(&mut pass, &self.blend_pipe, &self.globals_bg, &self.vbuf, ranges[2].clone());
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
struct Globals { size: vec2<f32>, pad: vec2<f32> };
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
    }
    let alpha = i.color.a * a;
    return vec4<f32>(i.color.rgb * alpha, alpha);
}
"#;

const PHOSPHOR_WGSL: &str = r#"
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

// glow' = blur(glow) · keep + light that left this pixel
@fragment
fn update(i: VOut) -> @location(0) vec4<f32> {
    let o = p.texel;
    let blurred = textureSample(glow, smp, i.uv).rgb * 0.5
        + (textureSample(glow, smp, i.uv + vec2<f32>(o.x, 0.0)).rgb
         + textureSample(glow, smp, i.uv - vec2<f32>(o.x, 0.0)).rgb
         + textureSample(glow, smp, i.uv + vec2<f32>(0.0, o.y)).rgb
         + textureSample(glow, smp, i.uv - vec2<f32>(0.0, o.y)).rgb) * 0.125;
    let lost = max(textureSample(live, smp, i.uv).rgb - textureSample(live_now, smp, i.uv).rgb, vec3<f32>(0.0));
    return vec4<f32>(blurred * p.keep + lost, 1.0);
}

// A texture copied (and rescaled) as it is.
@fragment
fn copy(i: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(glow, smp, i.uv).rgb, 1.0);
}

// shown = live + glow, rolled off softly above 0.8 so bright overlaps
// compress instead of clipping.
@fragment
fn composite(i: VOut) -> @location(0) vec4<f32> {
    let c = textureSample(glow, smp, i.uv).rgb + textureSample(live, smp, i.uv).rgb;
    let knee = p.knee;
    let over = max(c - vec3<f32>(knee), vec3<f32>(0.0));
    let soft = vec3<f32>(knee) + (1.0 - knee) * (vec3<f32>(1.0) - exp(-over / (1.0 - knee)));
    return vec4<f32>(select(c, soft, c > vec3<f32>(knee)), 1.0);
}
"#;
