//! Read the frontmost app's selected text.
//!
//! macOS uses the Accessibility API (`AXSelectedText` on the focused UI
//! element), so Read Aloud never touches the clipboard. Other platforms keep
//! the copy-and-restore capture, which still only returns an actual selection
//! and never falls back to clipboard contents.

use tauri::AppHandle;

/// The current selection, or `None` when nothing is selected or the focused
/// app does not expose it.
pub fn selected_text(app: &AppHandle) -> Result<Option<String>, String> {
    #[cfg(target_os = "macos")]
    {
        if !macos::is_trusted() {
            // The dev build (or a first run before the permission is granted)
            // cannot read the Accessibility tree. Keep selection-only Read
            // Aloud usable by temporarily copying the selection; the clipboard
            // is still restored and never used as a fallback text source.
            log::debug!("Read Aloud: Accessibility not trusted; using copy capture");
            return crate::clipboard::capture_selected_text(app);
        }
        match macos::selected_text() {
            Some(text) if !text.trim().is_empty() => Ok(Some(text)),
            _ => Ok(None),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        crate::clipboard::capture_selected_text(app)
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use core_foundation::base::{CFTypeRef, TCFType};
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
