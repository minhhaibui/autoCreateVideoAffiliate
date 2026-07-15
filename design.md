# Design — MoneyPrinterTurbo WebUI

A locked design system for this app. Every page/section redesign reads this file
before emitting code. Do not regenerate per page — extend or amend this file when
the system needs to grow.

The WebUI is a **Streamlit app** (webui/Main.py, single page, 3-step wizard).
The system is delivered through two layers only:

1. `.streamlit/config.toml` `[theme]` — colours, radius, fonts (mirrored in
   `webui/.streamlit/config.toml` for users who launch from `webui/`).
2. One injected `<style>` block at the top of `webui/Main.py` — structural
   polish only, targeting stable `data-testid` / `data-baseweb` hooks.

No other CSS may be scattered through the file. Streamlit widgets stay stock.

## Genre

playful — the *soft app register* (friendly, warm, not exuberant). The audience
is a non-technical Vietnamese TikTok-affiliate creator; the tone is "the room is
warm and someone smart is helping", never childish.

## Macrostructure family

- App page (the only page): **Workbench** — a 3-step wizard (stepper tabs) with
  a persistent primary action footer. No enrichment, no hero, no mascot:
  function carries the page.

## Theme

Custom (Hum-adapted for an app tool). Single dominant accent — the multi-accent
exuberance of catalog-Hum is wrong for a work tool.

- `--mpt-paper`        `#FAF7EF`  oklch(97.5% 0.012 95) — warm cream, never pure white
- `--mpt-paper-2`      `#F2EDE0`  — widget/secondary surfaces
- `--mpt-paper-3`      `#EAE3D2`  — hover band
- `--mpt-ink`          `#23252C`  oklch(20% 0.012 250) — near-black, cool tilt, never pure black
- `--mpt-ink-2`        `#5B5E6A`  — muted ink (captions, hints)
- `--mpt-accent`       `#D64F35`  oklch(58% 0.17 30) — coral; owns the primary action only
- `--mpt-accent-deep`  `#A83A24`  — the button's solid edge (press feedback)
- `--mpt-accent-soft`  `#F9E2DB`  — selected stepper pill
- `--mpt-rule`         `#E5DECC`  — borders/rules
- link colour          `#0B7285`  — teal-cyan, links only

## Typography

- Display + body: **Plus Jakarta Sans** 400/500/600/700/800 (Google Fonts —
  ships a `vietnamese` unicode-range subset; Vietnamese diacritics are a hard
  requirement). Falls back to Segoe UI / system-ui.
- Mono: Streamlit default (code blocks, logs).
- Wordmark h1 is compact (~1.65rem, weight 800, tracking -0.02em) — the page
  is a tool, not a landing hero.
- No serif anywhere. No italic headings.

## Spacing & shape

- Streamlit's own spacing scale; don't fight it.
- `baseRadius 12px` (inputs/widgets) · cards & expanders 16px · buttons pill (999px).
- No square corners anywhere.

## Motion

- Buttons: "the press is the feedback" — hover lifts 1px, `:active` presses
  down 1px; primary button carries a solid `--mpt-accent-deep` edge shadow that
  shrinks on press. Easing `cubic-bezier(0.2, 0.7, 0.3, 1)`, ~140ms.
- Nothing else animates. `prefers-reduced-motion: reduce` disables transforms.

## Microinteractions stance

- Silent success (st.toast) — never celebratory.
- Errors stay inline next to the widget that caused them.

## CTA voice

- Exactly one primary (coral) button per screen: **▶ Tạo Video / Generate
  Video** in the persistent footer. Every other button is secondary style.
- Verbs, sentence case, Vietnamese-first copy via i18n keys.

## What every section MUST share

- The cream paper / warm rule / coral-accent palette above.
- Plus Jakarta Sans.
- Bordered `st.container` cards with caption hints as section framing.
- The stepper-pill tab language.

## What sections MAY differ on

- Internal layout (columns, expanders, sliders) — whatever the task needs.

## Never

- Pure white paper, pure black ink, purple gradients, emoji-as-iconography in
  chrome (emoji in copy/labels that already exist in i18n is fine), invented
  metrics, stacked coral buttons.
