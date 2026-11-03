"""Look of the app: a copy editor's blue pencil and yellow highlighter."""

HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@400;500;700'
    '&family=Literata:opsz,wght@7..72,400;7..72,600&display=swap" rel="stylesheet">'
)

CSS = """
:root, .dark {
  --paper: #eef1f7;
  --sheet: #ffffff;
  --ink: #16204f;
  --ink-soft: #4b5478;
  --pencil: #2746d8;
  --pencil-soft: #d8def8;
  --marker: #fff06a;
  --line: #c9d0e2;
  --slip: #fff06a;
  --slip-unsure: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:not(.light) {
    --paper: #141a36;
    --sheet: #1d2547;
    --ink: #eef1fb;
    --ink-soft: #b4bcd8;
    --pencil: #9db0ff;
    --pencil-soft: #333e70;
    --marker: #6b6400;
    --line: #3a4470;
    --slip: #fff06a;
    --slip-unsure: #1d2547;
  }
}
.dark {
  --paper: #141a36;
  --sheet: #1d2547;
  --ink: #eef1fb;
  --ink-soft: #b4bcd8;
  --pencil: #9db0ff;
  --pencil-soft: #333e70;
  --marker: #6b6400;
  --line: #3a4470;
  --slip-unsure: #1d2547;
}
.gradio-container {
  --body-background-fill: var(--paper);
  --background-fill-primary: var(--sheet);
  --background-fill-secondary: var(--paper);
  --block-background-fill: transparent;
  --block-border-width: 0px;
  --block-shadow: none;
  --block-padding: 0px;
  --panel-background-fill: transparent;
  --panel-border-width: 0px;
  --input-background-fill: var(--sheet);
  --input-border-color: var(--line);
  --input-border-width: 1px;
  --input-shadow: none;
  --input-radius: 3px;
  --border-color-primary: var(--line);
  --border-color-accent: var(--pencil);
  --color-accent: var(--pencil);
  --color-accent-soft: var(--pencil-soft);
  --checkbox-background-color-selected: var(--pencil);
  --body-text-color: var(--ink);
  --body-text-color-subdued: var(--ink-soft);
  --block-label-text-color: var(--ink-soft);
  --block-title-text-color: var(--ink);
  --block-label-background-fill: transparent;
  --block-title-background-fill: transparent;
  --block-label-text-size: 0.9375rem;
  --block-title-text-size: 0.9375rem;
  --block-title-text-weight: 500;
  --block-label-padding: 0 0 6px 0;
  --block-title-padding: 0 0 6px 0;
  --layout-gap: 16px;
  --tab-text-color-selected: var(--pencil);
}
.gradio-container .html-container, .gradio-container .prose { padding: 0 !important; }
.gradio-container .tab-container button { font-size: 1rem; font-family: inherit; color: var(--ink); }
.gradio-container .tab-container button.selected { color: var(--pencil); font-weight: 700; }
.gradio-container .tab-container .selected::after { background: var(--pencil) !important; }
body, gradio-app, .gradio-container {
  background: var(--paper) !important;
  color: var(--ink);
  font-family: "Familjen Grotesk", "Segoe UI", system-ui, sans-serif !important;
}
.gradio-container { --font-mono: "Familjen Grotesk", system-ui, sans-serif; box-sizing: border-box; width: 100% !important; min-width: 0 !important; max-width: 1120px !important; margin: 0 auto !important; padding: 32px clamp(16px, 4vw, 24px) 64px !important; }
.gradio-container .form { background: transparent !important; border: 0 !important; }
footer { display: none !important; }

.masthead { margin: 0 0 28px; max-width: 34em; }
.masthead h1 {
  font-size: clamp(2.2rem, 6vw, 3.6rem); line-height: 1.02; letter-spacing: -0.03em;
  font-weight: 700; margin: 0 0 12px; color: var(--ink);
}
.masthead p { font-size: 1.125rem; line-height: 1.45; margin: 0; color: var(--ink-soft); }

.workbench { gap: 28px !important; align-items: flex-start !important; }

.story-box, .story-box > *, .story-box label, .story-box .wrap { background: transparent !important; border: 0 !important; box-shadow: none !important; padding: 0 !important; }
.story-box textarea {
  font-family: "Literata", Georgia, serif !important; font-size: 1.0625rem !important; line-height: 1.65 !important;
  color: var(--ink) !important; background: var(--sheet) !important;
  border: 1px solid var(--line) !important; border-radius: 3px !important; padding: 20px 22px !important;
  box-shadow: none !important;
}
.story-box textarea:focus { border-color: var(--pencil) !important; outline: 2px solid var(--pencil) !important; outline-offset: 1px; }
.story-box textarea::placeholder { color: var(--ink-soft) !important; opacity: 1; }

.controls { gap: 12px !important; align-items: flex-end !important; }
button.route {
  background: var(--pencil) !important; color: var(--sheet) !important; border: 0 !important;
  border-radius: 3px !important; font-weight: 700 !important; font-size: 1.0625rem !important;
  padding: 13px 22px !important; box-shadow: none !important; font-family: inherit !important;
}
button.route:hover { filter: brightness(1.12); }
button.quiet {
  background: transparent !important; color: var(--ink) !important; border: 1px solid var(--line) !important;
  border-radius: 3px !important; box-shadow: none !important; font-family: inherit !important; font-weight: 500 !important;
}
button:focus-visible { outline: 2px solid var(--pencil) !important; outline-offset: 2px; }

.slip {
  position: relative; background: var(--slip); color: #16204f; padding: 24px 24px 26px;
  transform: rotate(-1.1deg); transform-origin: top left;
  clip-path: polygon(0 0, 100% 0, 100% calc(100% - 10px), 96% 100%, 92% calc(100% - 10px), 88% 100%, 84% calc(100% - 10px), 80% 100%, 76% calc(100% - 10px), 72% 100%, 68% calc(100% - 10px), 64% 100%, 60% calc(100% - 10px), 56% 100%, 52% calc(100% - 10px), 48% 100%, 44% calc(100% - 10px), 40% 100%, 36% calc(100% - 10px), 32% 100%, 28% calc(100% - 10px), 24% 100%, 20% calc(100% - 10px), 16% 100%, 12% calc(100% - 10px), 8% 100%, 4% calc(100% - 10px), 0 100%);
}
.slip-unsure { background: var(--slip-unsure); color: var(--ink); outline: 2px dashed var(--pencil); outline-offset: -2px; clip-path: none; }
.slip-empty { background: var(--pencil-soft); color: var(--ink); }
.slip, .slip * { color: #16204f !important; }
.slip-unsure, .slip-unsure *, .slip-empty, .slip-empty * { color: var(--ink) !important; }
.slip p { margin: 0; }
.slip-kicker { font-size: 1rem; font-weight: 500; }
.slip-desk { font-size: clamp(1.9rem, 4vw, 2.6rem); line-height: 1.04; letter-spacing: -0.025em; font-weight: 700; margin: 4px 0 8px !important; overflow-wrap: anywhere; }
.slip-note { font-size: 1rem; line-height: 1.4; }
.odds-list { margin-top: 18px; display: grid; gap: 12px; }
.odds { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; font-size: 0.9375rem; }
.odds.strong { font-weight: 700; }
.odds-track { grid-column: 1 / -1; height: 6px; background: rgba(22, 32, 79, 0.14); border-radius: 0 3px 3px 0; overflow: hidden; }
.slip-unsure .odds-track { background: var(--pencil-soft); }
.odds-fill { display: block; height: 100%; background: currentColor; border-radius: 0 3px 3px 0; }

.story { margin-top: 8px; background: var(--sheet); border: 1px solid var(--line); border-radius: 3px; padding: 20px 22px; color: var(--ink); }
.story-text { font-family: "Literata", Georgia, serif; font-size: 1.0625rem; line-height: 1.75; max-width: 66ch; white-space: pre-wrap; overflow-wrap: anywhere; }
.mark-key { display: flex; flex-wrap: wrap; gap: 6px 22px; margin: 0 0 14px; font-size: 0.9375rem; color: var(--ink-soft); }
mark.for { background: var(--marker); color: var(--ink); padding: 0 2px; border-radius: 2px; }
mark.against { background: transparent; color: var(--ink); text-decoration: underline wavy var(--pencil) 2px; text-underline-offset: 3px; }
.unread-note { margin: 14px 0 0; padding-top: 12px; border-top: 1px dashed var(--line); color: var(--ink-soft); font-size: 0.9375rem; }

.verdict, .verdict > * { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.verdict { display: flex; flex-direction: column; gap: 12px; padding: 0 !important; }
.verdict label { color: var(--ink); }
.verdict-status { min-height: 1.4em; color: var(--ink-soft); }
.about { max-width: 68ch; color: var(--ink); font-size: 1.0625rem; line-height: 1.55; }
.about h2 { font-size: 1.5rem; letter-spacing: -0.015em; margin: 28px 0 8px; }
.about h2:first-child { margin-top: 0; }
.about table { border-collapse: collapse; width: 100%; font-size: 1rem; }
.about table, .about th, .about td { border: 0 !important; background: transparent !important; color: var(--ink) !important; }
.about th, .about td { text-align: left; padding: 7px 12px 7px 0 !important; border-bottom: 1px solid var(--line) !important; }
.about td.number, .about th.number { text-align: right; font-variant-numeric: tabular-nums; padding-right: 0 !important; }

@media (max-width: 760px) {
  .workbench { flex-direction: column !important; }
  .slip { transform: none; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""
