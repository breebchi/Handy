//! Read the frontmost app's selected text without modifying the clipboard.

/// The current selection, when available through a native read-only API.
pub fn selected_text() -> Option<String> {
    if crate::secure_input::is_enabled_now() {
        return None;
    }
    #[cfg(target_os = "macos")]
    if macos::is_trusted() {
        return macos::selected_text();
    }
    None
}

#[cfg(target_os = "macos")]
mod macos {
    use core_foundation::base::{CFGetTypeID, CFTypeRef, TCFType};
    use core_foundation::string::{CFString, CFStringRef};
    use std::ffi::c_void;

    type AXUIElementRef = *mut c_void;

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXUIElementCreateSystemWide() -> AXUIElementRef;
        fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> i32;
        fn AXIsProcessTrusted() -> bool;
        fn CFRelease(value: CFTypeRef);
    }

    pub(super) fn is_trusted() -> bool {
        unsafe { AXIsProcessTrusted() }
    }

    pub(super) fn selected_text() -> Option<String> {
        unsafe {
            let system = AXUIElementCreateSystemWide();
            if system.is_null() {
                return None;
            }

            let focused_attr = CFString::new("AXFocusedUIElement");
            let mut focused: CFTypeRef = std::ptr::null();
            let status = AXUIElementCopyAttributeValue(
                system,
                focused_attr.as_concrete_TypeRef(),
                &mut focused,
            );
            CFRelease(system as CFTypeRef);
            if status != 0 || focused.is_null() {
                return None;
            }

            let selected_attr = CFString::new("AXSelectedText");
            let mut selected: CFTypeRef = std::ptr::null();
            let status = AXUIElementCopyAttributeValue(
                focused as AXUIElementRef,
                selected_attr.as_concrete_TypeRef(),
                &mut selected,
            );
            CFRelease(focused);
            if status != 0 || selected.is_null() {
                return None;
            }

            if CFGetTypeID(selected) != CFString::type_id() {
                CFRelease(selected);
                return None;
            }
            // `wrap_under_create_rule` takes ownership and releases the CFString
            // when it drops.
            let text = CFString::wrap_under_create_rule(selected as CFStringRef).to_string();
            if text.trim().is_empty() {
                None
            } else {
                Some(text)
            }
        }
    }
}
