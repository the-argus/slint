// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-1.1 OR LicenseRef-Slint-commercial

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use i_slint_core::api::PhysicalSize as PhysicalWindowSize;

use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::AttachmentImage;
use vulkano::image::{ImageAccess, ImageViewAbstract};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::{Handle, VulkanLibrary, VulkanObject};

// must be nonzero
const FRAMES_IN_FLIGHT: u8 = 3;

/// This surface renders into the given window using Vulkan.
pub struct VulkanSurface {
    resize_event: Cell<Option<PhysicalWindowSize>>,
    gr_context: RefCell<skia_safe::gpu::DirectContext>,
    // must be vulkano::format::Format::B8G8R8A8_UNORM
    images: RefCell<Vec<Arc<AttachmentImage>>>,
    image_views: RefCell<Vec<Arc<ImageView<AttachmentImage>>>>,
    instance_handle: ash::vk::Instance,
    device_handle: ash::vk::PhysicalDevice,
    frame_index: RefCell<Option<usize>>,
    memory_allocator: RefCell<StandardMemoryAllocator>,
}

impl VulkanSurface {
    /// Creates a Skia Vulkan rendering surface from the given Vukano device, queue family index,
    /// and size.
    pub fn from_resources(
        physical_device: Arc<PhysicalDevice>,
        queue_family_index: u32,
        size: PhysicalWindowSize,
    ) -> Result<Self, i_slint_core::platform::PlatformError> {
        /*
        eprintln!(
            "Vulkan device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );*/

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: DeviceExtensions::empty(),
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(|dev_err| format!("Failed to create suitable logical Vulkan device: {dev_err}"))?;
        let queue = queues.next().ok_or_else(|| format!("Not Vulkan device queue found"))?;

        let instance = physical_device.instance();
        let library = instance.library();

        let get_proc = |of| unsafe {
            let result = match of {
                skia_safe::gpu::vk::GetProcOf::Instance(instance, name) => {
                    library.get_instance_proc_addr(ash::vk::Instance::from_raw(instance as _), name)
                }
                skia_safe::gpu::vk::GetProcOf::Device(device, name) => {
                    (instance.fns().v1_0.get_device_proc_addr)(
                        ash::vk::Device::from_raw(device as _),
                        name,
                    )
                }
            };

            match result {
                Some(f) => f as _,
                None => {
                    //println!("resolve of {} failed", of.name().to_str().unwrap());
                    core::ptr::null()
                }
            }
        };

        let instance_handle = instance.handle();

        let backend_context = unsafe {
            skia_safe::gpu::vk::BackendContext::new(
                instance_handle.as_raw() as _,
                physical_device.handle().as_raw() as _,
                device.handle().as_raw() as _,
                (queue.handle().as_raw() as _, queue.id_within_family() as _),
                &get_proc,
            )
        };

        let gr_context = skia_safe::gpu::DirectContext::new_vulkan(&backend_context, None)
            .ok_or_else(|| format!("Error creating Skia Vulkan context"))?;

        let mut images = Vec::<Arc<AttachmentImage>>::with_capacity(FRAMES_IN_FLIGHT as usize);
        let mut image_views =
            Vec::<Arc<ImageView<AttachmentImage>>>::with_capacity(FRAMES_IN_FLIGHT as usize);

        // NOTE: free list allocator, which can potentially lead to external
        // fragmentation. not likely for this usecase, but see
        // https://docs.rs/vulkano/latest/vulkano/memory/allocator/suballocator/struct.FreeListAllocator.html
        // if performance becomes a problem.
        // PoolAllocator would be ideal except I believe it requires compiletime known block sizes
        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

        Self::recreate_size_dependent_resources(
            size,
            &memory_allocator,
            &mut images,
            &mut image_views,
        )?;

        Ok(Self {
            resize_event: Cell::new(size.into()),
            gr_context: RefCell::new(gr_context),
            images: RefCell::new(images),
            image_views: RefCell::new(image_views),
            instance_handle,
            device_handle: physical_device.handle(),
            frame_index: RefCell::new(None),
            memory_allocator: RefCell::new(memory_allocator),
        })
    }

    pub fn recreate_size_dependent_resources(
        size: PhysicalWindowSize,
        memory_allocator: &StandardMemoryAllocator,
        output_images: &mut Vec<Arc<AttachmentImage>>,
        output_image_views: &mut Vec<Arc<ImageView<AttachmentImage>>>,
    ) -> Result<(), i_slint_core::platform::PlatformError> {
        for _ in 0..FRAMES_IN_FLIGHT {
            let image = AttachmentImage::new(
                memory_allocator,
                [size.width, size.height],
                Format::B8G8R8A8_UNORM,
            )
            .map_err(|vke| format!("Failed to create render target image: {vke}"))?;

            let image_view = ImageView::new_default(image.clone())
                .map_err(|vke| format!("Failed to create image view from image: {vke}"))?;

            output_images.push(image);
            output_image_views.push(image_view);
        }
        Ok(())
    }

    pub fn raw_vulkan_instance_handle(&self) -> u64 {
        return self.instance_handle.as_raw();
    }

    pub fn raw_vulkan_physical_device_handle(&self) -> u64 {
        return self.device_handle.as_raw();
    }

    pub fn current_raw_offscreen_vulkan_image_handle(&self) -> u64 {
        self.images.clone().take()[self.current_vulkan_frame_index()]
            .inner()
            .image
            .handle()
            .as_raw()
    }

    fn current_vulkan_frame_index(&self) -> usize {
        match self.frame_index.clone().take() {
            Some(idx) => idx,
            None => panic!("Vulkan frame index requested before first render"),
        }
    }
}

impl super::Surface for VulkanSurface {
    fn new(
        _window_handle: raw_window_handle::WindowHandle<'_>,
        _display_handle: raw_window_handle::DisplayHandle<'_>,
        size: PhysicalWindowSize,
    ) -> Result<Self, i_slint_core::platform::PlatformError> {
        let library = VulkanLibrary::new()
            .map_err(|load_err| format!("Error loading vulkan library: {load_err}"))?;

        let required_extensions = InstanceExtensions {
            khr_get_physical_device_properties2: true,
            ..InstanceExtensions::empty()
        }
        .intersection(library.supported_extensions());

        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                enumerate_portability: true,
                ..Default::default()
            },
        )
        .map_err(|instance_err| format!("Error creating Vulkan instance: {instance_err}"))?;

        let device_extensions = DeviceExtensions::empty();
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .map_err(|vke| format!("Error enumerating physical Vulkan devices: {vke}"))?
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(_, q)| q.queue_flags.intersects(QueueFlags::GRAPHICS))
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .ok_or_else(|| format!("Vulkan: Failed to find suitable physical device"))?;

        Self::from_resources(physical_device, queue_family_index, size)
    }

    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn resize_event(
        &self,
        _size: PhysicalWindowSize,
    ) -> Result<(), i_slint_core::platform::PlatformError> {
        self.resize_event.set(_size.into());
        Ok(())
    }

    fn render(
        &self,
        _size: PhysicalWindowSize,
        callback: &dyn Fn(&mut skia_safe::Canvas, &mut skia_safe::gpu::DirectContext),
    ) -> Result<(), i_slint_core::platform::PlatformError> {
        let gr_context = &mut self.gr_context.borrow_mut();

        let frame_index = match self.frame_index.clone().take() {
            Some(idx) => idx,
            None => 0,
        };

        let resize = self.resize_event.take();

        if resize.is_some() {
            let mut new_images =
                Vec::<Arc<AttachmentImage>>::with_capacity(FRAMES_IN_FLIGHT as usize);
            let mut new_image_views =
                Vec::<Arc<ImageView<AttachmentImage>>>::with_capacity(FRAMES_IN_FLIGHT as usize);

            VulkanSurface::recreate_size_dependent_resources(
                resize.unwrap(),
                &self.memory_allocator.borrow(),
                &mut new_images,
                &mut new_image_views,
            )?;

            *self.images.borrow_mut() = new_images;
            *self.image_views.borrow_mut() = new_image_views;
        }

        let images = self.images.borrow();

        if images.is_empty() {
            return Ok(());
        }

        let dim = images[frame_index].dimensions();

        let image_view = self.image_views.borrow()[frame_index].clone();
        let image_object = image_view.as_ref().image();
        let format = image_view.as_ref().format();

        debug_assert_eq!(format, Some(vulkano::format::Format::B8G8R8A8_UNORM));
        let (vk_format, color_type) =
            (skia_safe::gpu::vk::Format::B8G8R8A8_UNORM, skia_safe::ColorType::BGRA8888);

        let alloc = skia_safe::gpu::vk::Alloc::default();
        let image_info = &unsafe {
            skia_safe::gpu::vk::ImageInfo::new(
                image_object.inner().image.handle().as_raw() as _,
                alloc,
                skia_safe::gpu::vk::ImageTiling::OPTIMAL,
                skia_safe::gpu::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk_format,
                1,
                None,
                None,
                None,
                None,
            )
        };

        let render_target = &skia_safe::gpu::BackendRenderTarget::new_vulkan(
            (dim.width() as _, dim.height() as _),
            0,
            image_info,
        );

        let mut skia_surface = skia_safe::gpu::surfaces::wrap_backend_render_target(
            gr_context,
            render_target,
            skia_safe::gpu::SurfaceOrigin::TopLeft,
            color_type,
            None,
            None,
        )
        .ok_or_else(|| format!("Error creating Skia Vulkan surface"))?;

        callback(skia_surface.canvas(), gr_context);

        drop(skia_surface);

        // NOTE: evil. sync cpu, meaning wait until the GPU has finished rendering
        // to the image. to make this work for real there needs to be a way of
        // adding a fence signal to the queue submission which is hidden deep
        // in skia
        gr_context.submit(true);

        let new_frame_index = (frame_index + 1) % FRAMES_IN_FLIGHT as usize;
        *self.frame_index.borrow_mut() = Some(new_frame_index);

        Ok(())
    }

    fn bits_per_pixel(&self) -> Result<u8, i_slint_core::platform::PlatformError> {
        Ok(32)
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
