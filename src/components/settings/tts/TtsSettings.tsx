import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { listen } from "@tauri-apps/api/event";
import { confirm } from "@tauri-apps/plugin-dialog";
import { AudioLines, Download, ExternalLink, Trash2 } from "lucide-react";
import { commands, type TtsStatus } from "@/bindings";
import { notify } from "@/stores/noticeStore";
import { useSettings } from "../../../hooks/useSettings";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { ToggleSwitch } from "../../ui/ToggleSwitch";
import { Slider } from "../../ui/Slider";
import { Dropdown } from "../../ui/Dropdown";
import { Button } from "../../ui/Button";
import { ShortcutInput } from "../ShortcutInput";

const VOICE_LABELS: Record<string, string> = {
  F1: "settings.readAloud.voices.f1",
  F2: "settings.readAloud.voices.f2",
  F3: "settings.readAloud.voices.f3",
  F4: "settings.readAloud.voices.f4",
  F5: "settings.readAloud.voices.f5",
  M1: "settings.readAloud.voices.m1",
  M2: "settings.readAloud.voices.m2",
  M3: "settings.readAloud.voices.m3",
  M4: "settings.readAloud.voices.m4",
  M5: "settings.readAloud.voices.m5",
};

const LANGUAGES = [
  "na",
  "ar",
  "bg",
  "cs",
  "da",
  "de",
  "el",
  "en",
  "es",
  "et",
  "fi",
  "fr",
  "hi",
  "hr",
  "hu",
  "id",
  "it",
  "ja",
  "ko",
  "lt",
  "lv",
  "nl",
  "pl",
  "pt",
  "ro",
  "ru",
  "sk",
  "sl",
  "sv",
  "tr",
  "uk",
  "vi",
] as const;

const formatBytes = (bytes: number) => {
  if (bytes < 1_000_000) return `${Math.max(0, Math.round(bytes / 1_000))} KB`;
  return `${(bytes / 1_000_000).toFixed(0)} MB`;
};

export const TtsSettings: React.FC = () => {
  const { t, i18n } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const [status, setStatus] = useState<TtsStatus | null>(null);
  const [startingDownload, setStartingDownload] = useState(false);

  const enabled = getSetting("tts_enabled") ?? false;
  const agentEnabled = getSetting("tts_agent_enabled") ?? false;
  const voice = getSetting("tts_voice") ?? "F4";
  const language = getSetting("tts_language") ?? "en";
  const speed = getSetting("tts_speed") ?? 1.0;
  const volume = getSetting("tts_volume") ?? 0.9;
  const licenseAccepted = getSetting("tts_license_accepted") ?? false;

  const refreshStatus = useCallback(async () => {
    const result = await commands.getTtsStatus();
    if (result.status === "ok") {
      setStatus(result.data);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    // The 401 MB download is the only thing that changes quickly; poll often
    // while it runs and slowly otherwise.
    const interval = window.setInterval(
      () => void refreshStatus(),
      status?.model_downloading ? 500 : 3000,
    );
    return () => window.clearInterval(interval);
  }, [refreshStatus, status?.model_downloading]);

  useEffect(() => {
    const unlisten = listen<TtsStatus>("tts-status", ({ payload }) => {
      setStatus(payload);
    });
    return () => {
      void unlisten.then((dispose) => dispose());
    };
  }, []);

  const languageOptions = useMemo(() => {
    let displayNames: Intl.DisplayNames | null = null;
    try {
      displayNames = new Intl.DisplayNames([i18n.language], {
        type: "language",
      });
    } catch {
      displayNames = null;
    }
    return LANGUAGES.map((code) => ({
      value: code,
      label:
        code === "na"
          ? t("settings.readAloud.language.auto")
          : (displayNames?.of(code) ?? code.toUpperCase()),
    }));
  }, [i18n.language, t]);

  const voiceOptions = useMemo(
    () =>
      Object.keys(VOICE_LABELS).map((id) => ({
        value: id,
        label: t(VOICE_LABELS[id]),
      })),
    [t],
  );

  const startDownload = async () => {
    setStartingDownload(true);
    const result = await commands.downloadTtsModel();
    setStartingDownload(false);
    if (
      result.status === "error" &&
      !result.error.toLowerCase().includes("cancelled")
    ) {
      notify({
        title: t("settings.readAloud.model.downloadFailed"),
        detail: result.error,
      });
    }
    await refreshStatus();
  };

  const cancelDownload = async () => {
    await commands.cancelTtsDownload();
    await refreshStatus();
  };

  const deleteModel = async () => {
    const confirmed = await confirm(
      t("settings.readAloud.model.deleteConfirm"),
      {
        title: t("settings.readAloud.model.delete"),
        kind: "warning",
      },
    );
    if (!confirmed) {
      return;
    }
    const result = await commands.deleteTtsModel();
    if (result.status === "error") {
      notify({
        title: t("settings.readAloud.model.deleteFailed"),
        detail: result.error,
      });
    }
    await refreshStatus();
  };

  const preview = async () => {
    const result = await commands.previewTtsVoice(voice);
    if (result.status === "error") {
      notify({
        title: t("settings.readAloud.previewFailed"),
        detail: result.error,
      });
    }
  };

  const downloaded = status?.model_downloaded ?? false;
  const downloading = status?.model_downloading ?? false;
  const percent =
    status && status.total_bytes > 0
      ? Math.min(
          100,
          Math.round((status.downloaded_bytes / status.total_bytes) * 100),
        )
      : 0;

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.readAloud.title")}>
        <ToggleSwitch
          checked={enabled}
          onChange={(value) => void updateSetting("tts_enabled", value)}
          isUpdating={isUpdating("tts_enabled")}
          label={t("settings.readAloud.enable.label")}
          description={t("settings.readAloud.enable.description")}
          descriptionMode="tooltip"
          grouped={true}
        />
        <ShortcutInput shortcutId="read_aloud" grouped={true} />
        <div className="px-4 py-3 space-y-3">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <AudioLines width={16} height={16} className="text-ink-3" />
                <p className="text-sm font-medium">
                  {t("settings.readAloud.model.name")}
                </p>
                {downloaded && (
                  <span className="text-xs px-1.5 py-0.5 rounded bg-logo-primary/15 text-logo-primary">
                    {t("settings.readAloud.model.ready")}
                  </span>
                )}
              </div>
              <p className="text-sm text-ink-3 mt-1">
                {t("settings.readAloud.model.description")}
              </p>
              <p className="text-xs text-ink-3 mt-1">
                {t("settings.readAloud.model.size", {
                  size: formatBytes(status?.total_bytes ?? 0),
                })}
              </p>
              <a
                href="https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE"
                target="_blank"
                rel="noreferrer"
                className="mt-2 inline-flex items-center gap-1 text-xs text-ink-3 hover:text-text underline"
              >
                {t("settings.readAloud.model.license")}
                <ExternalLink width={12} height={12} />
              </a>
            </div>

            {downloaded ? (
              <Button variant="danger-ghost" size="sm" onClick={deleteModel}>
                <span className="flex items-center gap-1.5">
                  <Trash2 width={14} height={14} />
                  {t("settings.readAloud.model.delete")}
                </span>
              </Button>
            ) : downloading ? (
              <Button variant="secondary" size="sm" onClick={cancelDownload}>
                {t("settings.readAloud.model.cancel")}
              </Button>
            ) : (
              <Button
                variant="primary"
                size="sm"
                onClick={() => void startDownload()}
                disabled={
                  !licenseAccepted || startingDownload || status === null
                }
              >
                <span className="flex items-center gap-1.5">
                  <Download width={14} height={14} />
                  {t("settings.readAloud.model.download")}
                </span>
              </Button>
            )}
          </div>

          {downloading && status && (
            <div className="space-y-1">
              <div className="h-1.5 w-full rounded-full bg-wash overflow-hidden">
                <div
                  className="h-full bg-logo-primary transition-[width] duration-300"
                  style={{ width: `${percent}%` }}
                />
              </div>
              <p className="text-xs text-ink-3">
                {t("settings.readAloud.model.progress", {
                  percent,
                  downloaded: formatBytes(status.downloaded_bytes),
                  total: formatBytes(status.total_bytes),
                })}
              </p>
            </div>
          )}

          {!downloaded && !downloading && (
            <label className="flex items-start gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                className="mt-0.5"
                checked={licenseAccepted}
                onChange={(event) =>
                  void updateSetting(
                    "tts_license_accepted",
                    event.target.checked,
                  )
                }
              />
              <span className="text-ink-3">
                {t("settings.readAloud.model.acceptLicense")}
              </span>
            </label>
          )}
        </div>
        <div className="px-4 py-2 flex items-center justify-between min-h-12">
          <p className="text-sm font-medium">
            {t("settings.readAloud.voice.label")}
          </p>
          <div className="flex items-center gap-2">
            <Dropdown
              options={voiceOptions}
              selectedValue={voice}
              onSelect={(value) => void updateSetting("tts_voice", value)}
              disabled={!downloaded}
            />
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void preview()}
              disabled={!downloaded || !enabled || status?.speaking}
            >
              {t("settings.readAloud.voice.preview")}
            </Button>
          </div>
        </div>

        <div className="px-4 py-2 flex items-center justify-between min-h-12">
          <p className="text-sm font-medium">
            {t("settings.readAloud.language.label")}
          </p>
          <Dropdown
            options={languageOptions}
            selectedValue={language}
            onSelect={(value) => void updateSetting("tts_language", value)}
            disabled={!downloaded}
          />
        </div>

        <Slider
          value={speed}
          onChange={(value) => void updateSetting("tts_speed", value)}
          min={0.5}
          max={2}
          step={0.05}
          label={t("settings.readAloud.speed.label")}
          description={t("settings.readAloud.speed.description")}
          descriptionMode="tooltip"
          grouped={true}
          formatValue={(value) => `${value.toFixed(2)}×`}
        />
      </SettingsGroup>

      {status?.agent_supported && (
        <SettingsGroup title={t("settings.readAloud.agent.title")}>
          <ToggleSwitch
            checked={agentEnabled}
            onChange={(value) => void updateSetting("tts_agent_enabled", value)}
            isUpdating={isUpdating("tts_agent_enabled")}
            label={t("settings.readAloud.agent.label")}
            description={t("settings.readAloud.agent.description")}
            grouped={true}
          />
        </SettingsGroup>
      )}

      <SettingsGroup title={t("settings.readAloud.playbackTitle")}>
        {status?.speaking && (
          <div className="px-4 py-2 flex justify-end">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void commands.stopSpeaking()}
            >
              {t("settings.readAloud.voice.stop")}
            </Button>
          </div>
        )}
        <Slider
          value={volume}
          onChange={(value) => void updateSetting("tts_volume", value)}
          min={0}
          max={1}
          step={0.05}
          label={t("settings.readAloud.volume.label")}
          description={t("settings.readAloud.volume.description")}
          descriptionMode="tooltip"
          grouped={true}
          formatValue={(value) => `${Math.round(value * 100)}%`}
        />
      </SettingsGroup>

      <p className="px-4 text-xs text-ink-3">
        {t("settings.readAloud.outputNote")}
      </p>
    </div>
  );
};
