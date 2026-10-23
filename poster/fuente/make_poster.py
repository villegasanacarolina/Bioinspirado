"""Genera el póster A0 (español e inglés) como HTML y lo imprime a PDF con Chromium."""
import asyncio, pathlib
from playwright.async_api import async_playwright

ROOT = pathlib.Path(__file__).parent
FONT = "/usr/share/texmf/fonts/opentype/public/tex-gyre"

T = {
"es": dict(
 title="Alta exactitud interna, desempeño externo al nivel del azar",
 subtitle="Validación externa preregistrada de la selección de electrodos con NSGA-II para detectar deterioro cognitivo leve (MCI) con EEG",
 authors="Jessica Milagros Cordero Rivera · Ana Carolina Torres Villegas · Salvador Miguel Hinojosa Cervantes",
 affil="Tecnológico de Monterrey, Campus Guadalajara",
 qr="Preregistro en OSF<br><b>osf.io/4enh8</b>",
 hero=[("75.1% → 45.8%", "exactitud interna de la réplica → exactitud balanceada en 34 personas nuevas"),
       ("0 de 5", "métodos superaron el azar en la prueba externa preregistrada"),
       ("94.4%", "de exactitud para saber de qué base viene una persona solo con su EEG")],
 s1="1 · Contexto",
 s1t=["El <b>MCI</b> suele preceder a la demencia. El EEG en reposo es barato y portátil; un dispositivo con <b>pocos electrodos</b> facilitaría el tamizaje.",
      "<b>Aljalal et al. (2024, Sci Rep)</b> reportaron hasta <b>95%</b> de exactitud con 5–7 electrodos elegidos por <b>NSGA-II</b>, en 24 personas, con validación LOSO.",
      "Pero <b>eligieron y evaluaron con las mismas personas</b>, no publicaron código y no hicieron prueba externa."],
 s2="2 · Pregunta",
 s2t="¿Ese 95% describe cómo funcionaría el método con <b>personas nuevas</b>, o cuánto se ajustó a las 24 que ya tenía?",
 sd="Datos (públicos, Isfahan)",
 dat=[("", "EEG_1", "EEG_2"), ("Uso", "desarrollo", "prueba única"), ("Personas", "24 (11 MCI)", "34 (18 MCI)"), ("Equipo", "19 canales, 256 Hz", "igual"), ("Análisis", "primeros 10 min", "primeros 10 min")],
 s3="3 · Diseño",
 flow=[("EEG_1 · 24 personas", "réplica independiente en Python + controles diagnósticos"),
       ("Congelar 5 métodos", "R (réplica) · M0 · M4 · M1 · M6 + montaje físico"),
       ("Preregistro en OSF", "hipótesis, reglas de decisión, SHA-256 de código y datos"),
       ("EEG_2 · 34 personas", "mismo centro y equipo; prueba única, sin ajustes")],
 s4="4 · Métodos comparados",
 meth=[("R", "Réplica del paper: DWT + entropía, 19 canales, NSGA-II, regla de máxima exactitud"),
       ("M0", "EEG limpio con ICA, sin Fp1/Fp2/F7/F8, potencia relativa por banda, LDA regularizado"),
       ("M4", "Como M0, 1 electrodo por zona (búsqueda exhaustiva de 180 combinaciones)"),
       ("M1", "Como M0, NSGA-II con la regla del paper (5 semillas)"),
       ("M6", "Como M0, PSO binario que penaliza cada electrodo (5 semillas)"),
       ("Físico", "M0 y M4 sin referencia promedio ni ICA: electrodos reales")],
 s5="5 · Desarrollo: de dónde salen las cifras altas",
 s5t=["<b>Fuga entre segmentos:</b> por persona, 50–57%; mezclando segmentos de una misma persona, 90–99%. El 89.7% del paper coincide con nuestro 89.8% mezclado.",
      "<b>Sesgo de selección:</b> NSGA-II llega a <b>72–79%</b> con etiquetas al azar.",
      "<b>Ojos:</b> Fp1 y F8, de los más elegidos en el paper, tienen 68% y 39% de señal ocular."],
 s6="6 · Prueba externa preregistrada",
 s6t="La réplica tenía 75.1% interno y obtuvo 45.8% en personas nuevas. Los otros cuatro métodos clasificaron a <b>las 34 personas como MCI</b>.",
 s7="7 · Hipótesis",
 hyp=[("H1", "La réplica cae fuera de su exactitud interna", "Confirmada", "ok"),
      ("H2", "M0 no es inferior a la réplica", "Trivial", "tr"),
      ("H3", "M4/M6 no inferiores a M1 con menos electrodos", "Trivial / no", "tr"),
      ("H4", "Montaje físico no inferior a ICA", "No concluyente", "no")],
 s7n="“Trivial”: se cumple por regla, pero ambos métodos predicen MCI para todos, así que no prueba que funcionen.",
 s8="8 · ¿Por qué fallaron?",
 s8t=["Las dos bases son del mismo centro, pero <b>EEG_2 está dominada por delta</b> (69% frente a 40%). Como en EEG_1 el MCI tenía algo más de delta, los modelos mandan a todos a MCI.",
      "El <b>ruido de la red eléctrica (50 Hz)</b> se asocia con el diagnóstico <b>en sentidos opuestos</b> en cada base: un confusor de adquisición.",
      "Ni el AUC (0.40–0.52) ni estandarizar cada base por separado revelaron señal."],
 s9="9 · Electrodos elegidos",
 s9t="Estables entre semillas, pero sin valor en personas nuevas: estabilidad no es validez.",
 s10="10 · Conclusiones",
 s10t=["Las exactitudes reportadas <b>no se transfieren</b> a personas nuevas.",
       "El <b>cambio de dominio</b> y los <b>confusores de adquisición</b> pesan más que la señal del MCI.",
       "Minimizar electrodos solo tiene sentido <b>después</b> de demostrar que la señal generaliza."],
 s11="11 · Protocolo mínimo que proponemos",
 rec=["Evaluar por persona, nunca por segmento", "Selección anidada dentro de la validación", "Comparar contra etiquetas revueltas",
      "Reportar AUC además de exactitud", "Revisar ojos y ruido de red por grupo", "Reportar el montaje físico",
      "Probar si las bases se distinguen", "Preregistrar una prueba externa única"],
 lim="<b>Limitaciones:</b> muestras pequeñas (potencia 0.84 solo si la exactitud real fuera 70%); no se pudo verificar que EEG_2 use criterios idénticos; se probó la mejor configuración de selección de canales del paper, no todas.",
 refs="Aljalal M. et al. Sci Rep 14:12483 (2024) · Kashefpoor M. et al. J Med Signals Sens 6:25 (2016); Biomed Signal Process Control 53:101559 (2019) · Varma & Simon, BMC Bioinformatics 7:91 (2006) · Saeb et al., GigaScience 6:gix019 (2017) · Deb et al., IEEE TEC 6:182 (2002)",
 figcap_dev="Misma señal y clasificadores: por persona (azul) frente a segmentos mezclados (gris); rombos: valores del paper.",
 figcap_main="Exactitud balanceada en EEG_2 con IC 95%. Zona azul: lo necesario para superar el azar.",
 figcap_diff="Diferencias pareadas con IC 90%. Margen de no inferioridad: −10 puntos (línea naranja).",
 figcap_conf="Cada punto es una persona. Las diferencias entre bases superan a las diferencias entre grupos.",
),
"en": dict(
 title="High internal accuracy, chance-level external performance",
 subtitle="A preregistered external validation of NSGA-II electrode selection for EEG-based detection of mild cognitive impairment (MCI)",
 authors="Jessica Milagros Cordero Rivera · Ana Carolina Torres Villegas · Salvador Miguel Hinojosa Cervantes",
 affil="Tecnológico de Monterrey, Campus Guadalajara, Mexico",
 qr="Preregistered on OSF<br><b>osf.io/4enh8</b>",
 hero=[("75.1% → 45.8%", "internal accuracy of the replication → balanced accuracy in 34 new people"),
       ("0 of 5", "methods beat chance in the preregistered external test"),
       ("94.4%", "accuracy in telling which dataset a person comes from, from EEG alone")],
 s1="1 · Background",
 s1t=["<b>MCI</b> often precedes dementia. Resting EEG is cheap and portable; a device with <b>few electrodes</b> would ease screening.",
      "<b>Aljalal et al. (2024, Sci Rep)</b> reported up to <b>95%</b> accuracy with 5–7 electrodes chosen by <b>NSGA-II</b>, in 24 people, with LOSO validation.",
      "But electrodes were <b>chosen and evaluated on the same people</b>, no code was released and there was no external test."],
 s2="2 · Question",
 s2t="Does that 95% describe how the method would work on <b>new people</b>, or how well it fitted the 24 it already had?",
 sd="Data (public, Isfahan)",
 dat=[("", "EEG_1", "EEG_2"), ("Role", "development", "single test"), ("People", "24 (11 MCI)", "34 (18 MCI)"), ("Equipment", "19 channels, 256 Hz", "same"), ("Analysed", "first 10 min", "first 10 min")],
 s3="3 · Design",
 flow=[("EEG_1 · 24 people", "independent Python replication + diagnostic controls"),
       ("Freeze 5 methods", "R (replication) · M0 · M4 · M1 · M6 + physical montage"),
       ("OSF preregistration", "hypotheses, decision rules, SHA-256 of code and data"),
       ("EEG_2 · 34 people", "same centre and equipment; single test, no tuning")],
 s4="4 · Methods compared",
 meth=[("R", "Paper replication: DWT + entropy, 19 channels, NSGA-II, highest-accuracy rule"),
       ("M0", "ICA-cleaned EEG, no Fp1/Fp2/F7/F8, relative band power, regularised LDA"),
       ("M4", "As M0, one electrode per zone (exhaustive search, 180 combinations)"),
       ("M1", "As M0, NSGA-II with the paper's rule (5 seeds)"),
       ("M6", "As M0, binary PSO penalising each electrode (5 seeds)"),
       ("Physical", "M0 and M4 without average reference or ICA: real electrodes")],
 s5="5 · Development: where high figures come from",
 s5t=["<b>Segment leakage:</b> person-wise, 50–57%; mixing segments of the same person, 90–99%. The paper's 89.7% matches our mixed 89.8%.",
      "<b>Selection bias:</b> NSGA-II reaches <b>72–79%</b> with random labels.",
      "<b>Eyes:</b> Fp1 and F8, among the paper's most selected, carry 68% and 39% ocular signal."],
 s6="6 · Preregistered external test",
 s6t="The replication had 75.1% internally and scored 45.8% on new people. The other four methods classified <b>all 34 people as MCI</b>.",
 s7="7 · Hypotheses",
 hyp=[("H1", "Replication falls below its internal accuracy", "Supported", "ok"),
      ("H2", "M0 not inferior to the replication", "Trivial", "tr"),
      ("H3", "M4/M6 not inferior to M1 with fewer electrodes", "Trivial / no", "tr"),
      ("H4", "Physical montage not inferior to ICA", "Inconclusive", "no")],
 s7n="“Trivial”: met by the rule, but both methods predict MCI for everyone, so it does not show that they work.",
 s8="8 · Why did they fail?",
 s8t=["Both datasets come from the same centre, but <b>EEG_2 is delta-dominated</b> (69% vs 40%). Since MCI had slightly more delta in EEG_1, the models send everyone to MCI.",
      "<b>Mains interference (50 Hz)</b> is associated with the diagnosis <b>in opposite directions</b> in each dataset: an acquisition confounder.",
      "Neither the AUC (0.40–0.52) nor standardising each dataset separately revealed a signal."],
 s9="9 · Selected electrodes",
 s9t="Stable across seeds, but useless on new people: stability is not validity.",
 s10="10 · Conclusions",
 s10t=["The reported accuracies <b>do not transfer</b> to new people.",
       "<b>Dataset shift</b> and <b>acquisition confounders</b> outweigh the MCI signal.",
       "Minimising electrodes only makes sense <b>after</b> showing that the signal generalises."],
 s11="11 · Minimum protocol we propose",
 rec=["Evaluate by person, never by segment", "Nested selection inside validation", "Compare against shuffled labels",
      "Report AUC as well as accuracy", "Check eyes and mains noise by group", "Report the physical montage",
      "Test whether datasets can be told apart", "Preregister a single external test"],
 lim="<b>Limitations:</b> small samples (power 0.84 only if true accuracy were 70%); identical criteria for EEG_2 could not be verified; we tested the paper's best channel-selection configuration, not all.",
 refs="Aljalal M. et al. Sci Rep 14:12483 (2024) · Kashefpoor M. et al. J Med Signals Sens 6:25 (2016); Biomed Signal Process Control 53:101559 (2019) · Varma & Simon, BMC Bioinformatics 7:91 (2006) · Saeb et al., GigaScience 6:gix019 (2017) · Deb et al., IEEE TEC 6:182 (2002)",
 figcap_dev="Same signal and classifiers: person-wise (blue) vs mixed segments (grey); diamonds: values reported by the paper.",
 figcap_main="Balanced accuracy on EEG_2 with 95% CI. Blue area: what was needed to beat chance.",
 figcap_diff="Paired differences with 90% CI. Non-inferiority margin: −10 points (orange line).",
 figcap_conf="Each dot is a person. Differences between datasets exceed differences between groups.",
)}

CSS = f"""
@font-face {{ font-family: H; src: url('file://{FONT}/texgyreheros-regular.otf'); }}
@font-face {{ font-family: H; font-weight: bold; src: url('file://{FONT}/texgyreheros-bold.otf'); }}
@font-face {{ font-family: HC; font-weight: bold; src: url('file://{FONT}/texgyreheroscn-bold.otf'); }}
@page {{ size: 841mm 1189mm; margin: 0; }}
:root {{ --ink:#0b0b0b; --ink2:#52514e; --blue:#2a78d6; --blue-soft:#eef4fc; --orange:#eb6834; --orange-soft:#fdf0ea; --rule:#d9d8d3; --bg:#ffffff; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ width: 841mm; height: 1189mm; background: var(--bg); color: var(--ink); font-family: H, sans-serif; }}
body {{ padding: 30mm 32mm 24mm; display: flex; flex-direction: column; gap: 16mm; font-size: 26pt; line-height: 1.38; }}
header {{ display: grid; grid-template-columns: 1fr 120mm; gap: 20mm; align-items: center; border-left: 7mm solid var(--blue); padding-left: 16mm; }}
h1 {{ font-family: HC, H, sans-serif; font-size: 96pt; line-height: 1.02; letter-spacing: -0.5pt; }}
.sub {{ font-size: 37pt; line-height: 1.22; margin-top: 8mm; color: var(--ink); }}
.auth {{ font-size: 28pt; margin-top: 10mm; font-weight: bold; }}
.aff {{ font-size: 25pt; color: var(--ink2); margin-top: 2mm; }}
.qr {{ text-align: center; font-size: 22pt; color: var(--ink2); }}
.qr img {{ width: 92mm; height: 92mm; display: block; margin: 0 auto 4mm; }}
.hero {{ display: grid; grid-template-columns: 1.25fr 1fr 1fr; gap: 12mm; }}
.tile {{ background: var(--blue-soft); border-radius: 6mm; padding: 10mm 12mm; }}
.tile:first-child {{ background: var(--orange-soft); }}
.tile .n {{ font-family: HC, H, sans-serif; font-weight: bold; font-size: 92pt; line-height: 1; color: var(--blue); }}
.tile:first-child .n {{ color: var(--orange); }}
.tile .l {{ font-size: 25pt; margin-top: 4mm; line-height: 1.3; }}
main {{ flex: 1; display: grid; grid-template-columns: 1fr 1.18fr 1.12fr; gap: 16mm; }}
.col {{ display: flex; flex-direction: column; gap: 11mm; }}
section h2 {{ font-family: HC, H, sans-serif; font-weight: bold; font-size: 40pt; color: var(--blue); border-bottom: 1.2mm solid var(--blue); padding-bottom: 2mm; margin-bottom: 5mm; }}
section p {{ margin-bottom: 4mm; }}
ul.b {{ list-style: none; }}
ul.b li {{ position: relative; padding-left: 9mm; margin-bottom: 4mm; }}
ul.b li::before {{ content: ""; position: absolute; left: 0; top: 4.2mm; width: 3.6mm; height: 3.6mm; background: var(--blue); border-radius: 50%; }}
.q {{ font-size: 31pt; line-height: 1.3; background: var(--orange-soft); border-left: 3mm solid var(--orange); padding: 7mm 9mm; }}
.flow {{ display: flex; flex-direction: column; gap: 0; }}
.step {{ border: 0.8mm solid var(--rule); border-radius: 4mm; padding: 5mm 8mm; background: #fafaf8; }}
.step b {{ display: block; font-size: 27pt; }}
.step span {{ font-size: 22pt; color: var(--ink2); }}
.step.reg {{ border-color: var(--orange); background: var(--orange-soft); }}
.step.test {{ border-color: var(--blue); background: var(--blue-soft); }}
.arrow {{ text-align: center; color: var(--ink2); font-size: 26pt; line-height: 1.1; }}
table.m {{ width: 100%; border-collapse: collapse; font-size: 22pt; }}
table.m td {{ padding: 2.6mm 3mm; border-bottom: 0.5mm solid var(--rule); vertical-align: top; }}
table.m td:first-child {{ font-weight: bold; white-space: nowrap; color: var(--blue); width: 30mm; }}
figure {{ margin: 2mm 0 3mm; }}
figure img {{ width: 100%; display: block; }}
figcaption {{ font-size: 20pt; color: var(--ink2); margin-top: 2mm; line-height: 1.3; }}
table.d {{ width: 100%; border-collapse: collapse; font-size: 23pt; }}
table.d td {{ padding: 2.4mm 3mm; border-bottom: 0.5mm solid var(--rule); }}
table.d tr:first-child td {{ font-weight: bold; color: var(--blue); border-bottom: 0.8mm solid var(--blue); }}
table.d td:first-child {{ color: var(--ink2); }}
table.h {{ width: 100%; border-collapse: collapse; font-size: 23pt; }}
table.h td {{ padding: 3mm; border-bottom: 0.5mm solid var(--rule); vertical-align: middle; }}
table.h td:first-child {{ font-weight: bold; width: 20mm; }}
.chip {{ display: inline-block; padding: 1.2mm 4mm; border-radius: 10mm; font-size: 20pt; font-weight: bold; white-space: nowrap; }}
.chip.ok {{ background: var(--blue); color: white; }}
.chip.tr {{ background: #ecebe7; color: var(--ink); }}
.chip.no {{ background: white; color: var(--ink2); border: 0.6mm solid var(--ink2); }}
.note {{ font-size: 20pt; color: var(--ink2); margin-top: 3mm; }}
.concl li {{ font-size: 27pt; }}
.rec {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3mm 8mm; font-size: 21.5pt; }}
.rec div {{ padding-left: 10mm; position: relative; }}
.rec div::before {{ content: "✓"; position: absolute; left: 0; color: var(--blue); font-weight: bold; }}
.lim {{ font-size: 21pt; color: var(--ink); background: #f5f5f2; padding: 6mm 8mm; border-radius: 4mm; }}
footer {{ font-size: 17pt; color: var(--ink2); border-top: 0.6mm solid var(--rule); padding-top: 4mm; }}
"""

def html(L, lang):
    a = ROOT / "assets"
    hero = "".join(f'<div class="tile"><div class="n">{n}</div><div class="l">{l}</div></div>' for n, l in L["hero"])
    flow = ""
    for i, (b, s) in enumerate(L["flow"]):
        cls = " reg" if i == 2 else (" test" if i == 3 else "")
        flow += f'<div class="step{cls}"><b>{b}</b><span>{s}</span></div>' + ('<div class="arrow">↓</div>' if i < 3 else "")
    meth = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in L["meth"])
    hyp = "".join(f'<tr><td>{h}</td><td>{t}</td><td><span class="chip {c}">{v}</span></td></tr>' for h, t, v, c in L["hyp"])
    li = lambda xs: "".join(f"<li>{x}</li>" for x in xs)
    rec = "".join(f"<div>{r}</div>" for r in L["rec"])
    dat = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in L["dat"])
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{L['title']}</title><style>{CSS}</style></head><body>
<header><div><h1>{L['title']}</h1><div class="sub">{L['subtitle']}</div><div class="auth">{L['authors']}</div><div class="aff">{L['affil']}</div></div>
<div class="qr"><img src="file://{a}/qr_osf.svg">{L['qr']}</div></header>
<div class="hero">{hero}</div>
<main>
 <div class="col">
  <section><h2>{L['s1']}</h2><ul class="b">{li(L['s1t'])}</ul></section>
  <section><h2>{L['s2']}</h2><p class="q">{L['s2t']}</p></section>
  <section><h2>{L['sd']}</h2><table class="d">{dat}</table></section>
  <section><h2>{L['s3']}</h2><div class="flow">{flow}</div></section>
  <section><h2>{L['s4']}</h2><table class="m">{meth}</table></section>
 </div>
 <div class="col">
  <section><h2>{L['s5']}</h2><figure><img src="file://{a}/{lang}_p_dev.svg"><figcaption>{L['figcap_dev']}</figcaption></figure><ul class="b">{li(L['s5t'])}</ul></section>
  <section><h2>{L['s6']}</h2><figure><img src="file://{a}/{lang}_p_main.svg"><figcaption>{L['figcap_main']}</figcaption></figure><p>{L['s6t']}</p></section>
  <section><h2>{L['s7']}</h2><table class="h">{hyp}</table><p class="note">{L['s7n']}</p><figure><img src="file://{a}/{lang}_p_diff.svg"><figcaption>{L['figcap_diff']}</figcaption></figure></section>
 </div>
 <div class="col">
  <section><h2>{L['s8']}</h2><figure><img src="file://{a}/{lang}_p_conf.svg"><figcaption>{L['figcap_conf']}</figcaption></figure><ul class="b">{li(L['s8t'])}</ul></section>
  <section><h2>{L['s9']}</h2><figure><img src="file://{a}/{lang}_p_heads.svg" style="width:82%;margin:0 auto"></figure><p>{L['s9t']}</p></section>
  <section class="concl"><h2>{L['s10']}</h2><ul class="b">{li(L['s10t'])}</ul></section>
  <section><h2>{L['s11']}</h2><div class="rec">{rec}</div></section>
  <p class="lim">{L['lim']}</p>
 </div>
</main>
<footer>{L['refs']}</footer>
</body></html>"""

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium" if pathlib.Path("/opt/pw-browsers/chromium").is_file() else None)
        for lang in ("es", "en"):
            f = ROOT / f"poster_{lang}.html"; f.write_text(html(T[lang], lang), encoding="utf-8")
            pg = await b.new_page(viewport={"width": 3179, "height": 4494})
            await pg.goto(f"file://{f}"); await pg.wait_for_timeout(800)
            h = await pg.evaluate("document.body.scrollHeight")
            print(lang, "alto del contenido (px):", h, "de 4494")
            await pg.pdf(path=str(ROOT / f"Poster_{lang}.pdf"), width="841mm", height="1189mm", print_background=True)
            await pg.screenshot(path=str(ROOT / f"preview_{lang}.png"), full_page=False)
        await b.close()
asyncio.run(main())
