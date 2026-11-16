//! Tauri commands for the local Read Aloud (text-to-speech) feature.

use crate::settings::{get_settings, write_settings};
use crate::tts::{TtsManager, TtsStatus, LANGUAGES, VOICES};
use std::sync::Arc;
use tauri::{AppHandle, Manager};

fn manager(app: &AppHandle) -> Result<Arc<TtsManager>, String> {
    app.try_state::<Arc<TtsManager>>()
        .map(|state| state.inner().clone())
        .ok_or_else(|| "The Read Aloud manager is not initialized.".to_string())
}

#[tauri::command]
#[specta::specta]
pub fn get_tts_status(app: AppHandle) -> Result<TtsStatus, String> {
    Ok(manager(&app)?.status())
}

#[tauri::command]
#[specta::specta]
pub fn get_tts_voices() -> Vec<String> {
    VOICES.iter().map(|voice| (*voice).to_string()).collect()
}

#[tauri::command]
#[specta::specta]
pub fn get_tts_languages() -> Vec<String> {
    LANGUAGES.iter().map(|lang| (*lang).to_string()).collect()
}

/// Download the ~401 MB Supertonic 3 model. The frontend calls this without
/// awaiting it and polls `get_tts_status` (plus the progress event) instead.
#[tauri::command]
#[specta::specta]
pub async fn download_tts_model(app: AppHandle) -> Result<(), String> {
    let settings = get_settings(&app);
    if !settings.tts_license_accepted {
        return Err("Accept the Supertonic 3 model license before downloading.".to_string());
    }
    manager(&app)?.download().await
}

#[tauri::command]
#[specta::specta]
pub fn cancel_tts_download(app: AppHandle) -> Result<(), String> {
    manager(&app)?.cancel_download();
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn delete_tts_model(app: AppHandle) -> Result<(), String> {
    manager(&app)?.delete_model().await
}

#[tauri::command]
#[specta::specta]
pub async fn speak_text(app: AppHandle, text: String) -> Result<(), String> {
    manager(&app)?.speak_with_settings(text).await
}

#[tauri::command]
#[specta::specta]
pub async fn preview_tts_voice(app: AppHandle, voice: String) -> Result<(), String> {
    manager(&app)?.preview(voice).await
}

#[tauri::command]
#[specta::specta]
pub fn stop_speaking(app: AppHandle) -> Result<(), String> {
    manager(&app)?.stop();
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_tts_enabled_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.tts_enabled = enabled;
    write_settings(&app, settings);
    if enabled {
        if let Ok(manager) = manager(&app) {
            manager.prewarm();
        }
    }
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_tts_voice_setting(app: AppHandle, voice: String) -> Result<(), String> {
    if !VOICES.contains(&voice.as_str()) {
        return Err(format!("Unknown voice '{voice}'."));
    }
    let mut settings = get_settings(&app);
    settings.tts_voice = voice;
    write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_tts_language_setting(app: AppHandle, language: String) -> Result<(), String> {
    if !LANGUAGES.contains(&language.as_str()) {
        return Err(format!("Unsupported language '{language}'."));
    }
    let mut settings = get_settings(&app);
    settings.tts_language = language;
    write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_tts_speed_setting(app: AppHandle, speed: f32) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.tts_speed = speed.clamp(0.5, 2.0);
    write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_tts_volume_setting(app: AppHandle, volume: f32) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.tts_volume = volume.clamp(0.0, 1.0);
    write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn set_tts_license_accepted(app: AppHandle, accepted: bool) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.tts_license_accepted = accepted;
    write_settings(&app, settings);
    Ok(())
}
