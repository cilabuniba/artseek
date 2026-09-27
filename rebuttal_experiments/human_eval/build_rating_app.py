"""Generate the rating page: one self-contained HTML file (images embedded).

For each painting (one per screen) the rater sees the reference material (title,
artist, date, key facts, encyclopaedia summary or translated ICCD fields; the
same reference given to the phi-4 judge) and three descriptions labelled A/B/C.
The letters are assigned by a seeded shuffle of rater code + item id, so they
differ across raters and paintings and are stable across reloads. Every
description is scored 0/1/2 on subject, figures, placement and evidence, with
the scale definitions on screen, and the rater picks the best description.

Answers are saved in the browser (localStorage). At the end the rater copies a
short transcript (or the full JSON) and sends it; collect_ratings.py decodes
it. A test mode keeps practice runs separate from real ones.

Usage:
    python rebuttal_experiments/human_eval/build_rating_app.py
"""

import base64
import json
from pathlib import Path

import click

HERE = Path(__file__).resolve().parent

# The three systems shown to the raters.
RATED = ("base", "artseek_multiquery", "gpt")

SCALE = [
    (2, "Specifically correct"),
    (1, "Right category, wrong specifics"),
    (0, "Wrong"),
]

AXES = [
    ("subject", "Subject",
     "Is the depicted event, story or scene correctly identified?",
     "Names the specific subject", "Right general category only",
     "States a subject the reference contradicts"),
    ("figures", "Figures",
     "Are the figures identified, with the attributes that identify them?",
     "Names them and cites attributes", "Generic only (“a man”, “a saint”)",
     "Names the wrong figures"),
    ("placement", "Placement",
     "Is the tradition, school and period right?",
     "Tradition and period both about right", "One right, or very vague",
     "Wrong tradition or wrong century"),
    ("evidence", "Evidence",
     "Are the named works, artists or sources real and relevant?",
     "Specific, real, genuinely relevant", "Real but loosely relevant, or vague",
     "Nothing specific, or works that do not exist"),
]


def load_answers(results_dir: Path) -> dict:
    out: dict = {}
    for s in RATED:
        p = results_dir / f"{s}_normalized.json"
        if not p.exists():
            p = results_dir / f"{s}.json"
        for r in json.loads(p.read_text()):
            out.setdefault(r["id"], {})[s] = (
                r.get("answer_normalized") or r.get("answer") or "")
    return out


def reference_block(sheet: dict, item: dict) -> dict:
    """Exactly what a rater is allowed to see: catalogued fact, never a
    retrieved document and never which system said what."""
    ref = {
        "title": sheet.get("title") or item.get("title"),
        "artist": sheet.get("artist") or item.get("artist"),
        "year": sheet.get("year") or item.get("year") or item.get("century"),
        "facts": list(sheet.get("key_facts") or []),
        "source": sheet.get("key_facts_source") or "",
    }
    if sheet.get("wikipedia_extract"):
        ref["summary"] = sheet["wikipedia_extract"]
        ref["summary_source"] = sheet.get("wikipedia_url") or "English Wikipedia"
    if sheet.get("iccd"):
        # English translation when available, otherwise the Italian.
        note = (sheet["iccd"].get("NSC") or {})
        val = note.get("value_en") or note.get("value")
        if val:
            ref["note"] = val
            ref["note_source"] = "ICCD catalogue record"
    return ref


def build_items(items_file: str, sheets_file: str, results_dir: Path) -> list:
    items = json.loads((HERE / "data" / items_file).read_text())
    sheets = {s["id"]: s for s in
              json.loads((HERE / "data" / sheets_file).read_text())}
    answers = load_answers(results_dir)
    out = []
    for it in items:
        img = (HERE / it["image"]).read_bytes()
        out.append({
            "id": it["id"],
            "tier": it["tier"],
            "image": "data:image/jpeg;base64," + base64.b64encode(img).decode(),
            "reference": reference_block(sheets.get(it["id"], {}), it),
            "answers": {s: answers.get(it["id"], {}).get(s, "") for s in RATED},
        })
    return out


CSS = """
:root{
  --paper:#f4f5f8; --card:#ffffff; --ink:#181b23; --ink-2:#414a5c;
  --ink-3:#6f7889; --line:#d6dae2; --line-2:#e8ebf0;
  --accent:#2f4b8c; --accent-soft:#e7ecf7;
  --pick:#8a6218; --pick-soft:#f7f0df;
  --test:#8c3b3b; --test-soft:#f8e8e8;
  --shadow:0 1px 2px rgba(20,24,32,.05),0 8px 24px rgba(20,24,32,.05);
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --paper:#11131a; --card:#181b22; --ink:#e8ebf1; --ink-2:#b2bac9;
  --ink-3:#818a9b; --line:#2b313c; --line-2:#212630;
  --accent:#8fabe4; --accent-soft:#1b2434;
  --pick:#d8ab5b; --pick-soft:#2a2318;
  --test:#e08b8b; --test-soft:#2c1a1a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
}}
:root[data-theme="dark"]{
  --paper:#11131a; --card:#181b22; --ink:#e8ebf1; --ink-2:#b2bac9;
  --ink-3:#818a9b; --line:#2b313c; --line-2:#212630;
  --accent:#8fabe4; --accent-soft:#1b2434;
  --pick:#d8ab5b; --pick-soft:#2a2318;
  --test:#e08b8b; --test-soft:#2c1a1a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  font-size:16px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1020px;margin:0 auto;padding:28px 20px 80px}
h1,h2,h3{font-family:"Source Serif 4",Georgia,serif;font-weight:600;
  margin:0;letter-spacing:-.01em;text-wrap:balance}
h1{font-size:1.9rem} h2{font-size:1.25rem} h3{font-size:1.02rem}
p{margin:0;color:var(--ink-2)}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.7rem;
  text-transform:uppercase;letter-spacing:.11em;color:var(--accent);font-weight:500}

.card{background:var(--card);border:1px solid var(--line);border-radius:4px;
  box-shadow:var(--shadow)}
.pad{padding:22px}
.stack{display:flex;flex-direction:column;gap:18px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}

input[type=text],select{font:inherit;padding:9px 11px;border:1px solid var(--line);
  border-radius:3px;background:var(--card);color:var(--ink);min-width:210px}
input[type=text]:focus,select:focus{outline:2px solid var(--accent);outline-offset:1px}
label.fld{display:flex;flex-direction:column;gap:5px;font-size:.82rem;color:var(--ink-3)}
button{font:inherit;padding:9px 17px;border-radius:3px;border:1px solid var(--accent);
  background:var(--accent);color:#fff;cursor:pointer;font-weight:500}
button.ghost{background:transparent;color:var(--accent)}
button:disabled{opacity:.45;cursor:not-allowed}
button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}

.bar{position:sticky;top:0;z-index:20;background:var(--paper);
  border-bottom:1px solid var(--line);padding:10px 0;margin-bottom:20px}
.bar .inner{max-width:1020px;margin:0 auto;padding:0 20px;
  display:flex;gap:14px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.prog{flex:1;min-width:150px;height:5px;background:var(--line-2);border-radius:3px;overflow:hidden}
.prog i{display:block;height:100%;background:var(--accent);transition:width .2s}
.testbanner{background:var(--test-soft);color:var(--test);border:1px solid var(--test);
  border-radius:3px;padding:5px 11px;font-size:.78rem;font-weight:600;
  font-family:"IBM Plex Mono",ui-monospace,monospace}

.split{display:grid;grid-template-columns:minmax(0,340px) minmax(0,1fr);gap:22px;align-items:start}
@media (max-width:880px){.split{grid-template-columns:1fr}}
.pic{width:100%;border-radius:3px;display:block;background:var(--line-2)}
.ref{font-size:.88rem;display:flex;flex-direction:column;gap:9px}
.ref .k{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.72rem;
  text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3)}
.ref ul{margin:0;padding-left:17px;display:flex;flex-direction:column;gap:4px}
.ref .sum{color:var(--ink-2);max-height:190px;overflow:auto;padding-right:6px}

.desc{border:1px solid var(--line);border-radius:4px;background:var(--card);
  margin-bottom:16px;overflow:hidden}
.desc header{display:flex;gap:10px;align-items:center;padding:11px 16px;
  border-bottom:1px solid var(--line-2);background:var(--accent-soft)}
.letter{font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:600;
  font-size:1rem;color:var(--accent)}
.body{padding:14px 16px;white-space:pre-wrap;font-size:.92rem;color:var(--ink)}
.axes{padding:4px 16px 16px;display:flex;flex-direction:column;gap:12px}
.ax{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:center}
.ax .lab{font-size:.85rem}
.ax .lab b{color:var(--ink);font-weight:600}
.ax .lab span{color:var(--ink-3);display:block;font-size:.78rem}
.opts{display:flex;gap:5px}
.opts input{position:absolute;opacity:0;width:0;height:0}
.opts label{min-width:34px;height:32px;display:grid;place-items:center;
  border:1px solid var(--line);border-radius:3px;cursor:pointer;font-size:.9rem;
  font-family:"IBM Plex Mono",ui-monospace,monospace;background:var(--card)}
.opts label:hover{border-color:var(--accent)}
.opts input:checked+label{background:var(--accent);border-color:var(--accent);
  color:#fff;font-weight:600}
.opts input:focus-visible+label{outline:2px solid var(--ink);outline-offset:1px}

.choice{border:1px solid var(--pick);background:var(--pick-soft);border-radius:4px;
  padding:15px 18px;display:flex;flex-direction:column;gap:10px}
.choice .opts label{min-width:52px}
.choice .opts input:checked+label{background:var(--pick);border-color:var(--pick)}
textarea{font:inherit;width:100%;min-height:62px;padding:9px 11px;border:1px solid var(--line);
  border-radius:3px;background:var(--card);color:var(--ink);resize:vertical}

.legend{font-size:.78rem;color:var(--ink-3);display:flex;gap:14px;flex-wrap:wrap}
.legend b{color:var(--ink-2);font-weight:600}
.done{background:var(--accent-soft);border-radius:3px;padding:16px 18px;font-size:.9rem}
.err{color:var(--test);font-size:.85rem}
.hide{display:none!important}
"""


def build_html(items: list) -> str:
    data = json.dumps({"items": items, "axes": AXES, "scale": SCALE,
                       "systems": list(RATED)}, ensure_ascii=False)
    return f"""<title>ArtSeek Painting Ratings</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{CSS}</style>

<div id="bar" class="bar hide"><div class="inner">
  <span class="mono" id="who" style="font-size:.8rem;color:var(--ink-3)"></span>
  <span id="testflag" class="testbanner hide">PRACTICE — not real data</span>
  <div class="prog"><i id="pbar" style="width:0%"></i></div>
  <span class="mono" id="pnum" style="font-size:.8rem;color:var(--ink-3)"></span>
</div></div>

<div class="wrap">
  <div id="intro" class="stack">
    <div class="eyebrow">ArtSeek study</div>
    <h1>Rating painting descriptions</h1>
    <p>You will see <b>15 paintings</b>. For each one, three assistants wrote a
    description. Your job is to score how <b>correct</b> each description is
    against the catalogue information shown beside the painting — not how well
    it is written. Fluent writing earns nothing.</p>
    <p>The three descriptions appear as <b>A</b>, <b>B</b> and <b>C</b> in a
    different order for every painting and every rater, so there is no pattern
    to find. Roughly 30&ndash;45 minutes. Your answers save automatically in this
    browser; you can close the tab and come back.</p>

    <div class="card pad stack" style="max-width:560px">
      <label class="fld">Your rater code
        <input type="text" id="code" placeholder="e.g. R1, R2, EXPERT" autocomplete="off">
      </label>
      <label class="fld">Your background
        <select id="role">
          <option value="">Select…</option>
          <option value="lay">General visitor (no specialist art training)</option>
          <option value="expert">Art historian / museum professional</option>
        </select>
      </label>
      <label class="fld">Session type
        <select id="mode">
          <option value="">Select…</option>
          <option value="real">Real session — my answers count</option>
          <option value="test">Practice / test — discard my answers</option>
        </select>
      </label>
      <div class="row"><button id="start" disabled>Start</button>
        <span class="err hide" id="introerr"></span></div>
    </div>
  </div>

  <div id="app" class="hide"></div>
  <div id="finish" class="hide stack"></div>
</div>

<script>
const DATA = {data};
const $ = (s,r=document)=>r.querySelector(s);

let S = {{code:"", role:"", mode:"", idx:0, answers:{{}}, started:""}};

// Practice runs get their own key so a trial can never overwrite real work.
const keyFor = (code,mode)=>`artseek_v3_${{mode}}_${{(code||"").toUpperCase()}}`;

function save(){{
  try{{ localStorage.setItem(keyFor(S.code,S.mode), JSON.stringify(S)); }}catch(e){{}}
}}
function load(code,mode){{
  try{{
    const raw = localStorage.getItem(keyFor(code,mode));
    return raw ? JSON.parse(raw) : null;
  }}catch(e){{ return null; }}
}}

// Per-rater, per-item letter assignment. Same rater + same item always gives
// the same order (so a reload is stable), different raters differ.
function hash(s){{ let h=2166136261; for(let i=0;i<s.length;i++){{ h^=s.charCodeAt(i); h=Math.imul(h,16777619); }} return h>>>0; }}
function orderFor(item){{
  const sys=[...DATA.systems];
  const h=hash(S.code.toUpperCase()+"|"+item.id);
  // Deterministic Fisher-Yates driven by the hash.
  let x=h;
  for(let i=sys.length-1;i>0;i--){{ x=(x*1664525+1013904223)>>>0; const j=x%(i+1); [sys[i],sys[j]]=[sys[j],sys[i]]; }}
  return sys;
}}

// Returns what is still unanswered for one painting (empty = complete).
function missingFor(it){{
  const a=S.answers[it.id];
  const letters=["A","B","C"].slice(0,DATA.systems.length);
  const gaps=[];
  letters.forEach(L=>{{
    DATA.axes.forEach(ax=>{{
      const v=a && a.scores && a.scores[L] ? a.scores[L][ax[0]] : undefined;
      if(v===undefined||v===null) gaps.push(`Description ${{L}} · ${{ax[1]}}`);
    }});
  }});
  if(!a || !a.best) gaps.push("Overall choice");
  return gaps;
}}
function itemDone(it){{ return missingFor(it).length===0; }}
function nDone(){{ return DATA.items.filter(itemDone).length; }}

function refHTML(r){{
  let h=`<div class="ref">`;
  h+=`<div><div class="k">Catalogued title</div><b>${{esc(r.title||"—")}}</b></div>`;
  if(r.artist) h+=`<div><div class="k">Artist</div>${{esc(r.artist)}}</div>`;
  if(r.year) h+=`<div><div class="k">Date</div>${{esc(String(r.year))}}</div>`;
  if(r.facts?.length){{
    h+=`<div><div class="k">Catalogue facts</div><ul>`+
       r.facts.map(f=>`<li>${{esc(f)}}</li>`).join("")+`</ul></div>`;
  }}
  if(r.summary) h+=`<div><div class="k">Encyclopaedia summary</div><div class="sum">${{esc(r.summary)}}</div></div>`;
  if(r.note) h+=`<div><div class="k">Catalogue note</div><div class="sum">${{esc(r.note)}}</div></div>`;
  return h+`</div>`;
}}
function esc(s){{ return String(s??"").replace(/[&<>"]/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c]); }}

function render(){{
  const it=DATA.items[S.idx];
  const order=orderFor(it);
  const letters=["A","B","C"].slice(0,order.length);
  const a=S.answers[it.id]||(S.answers[it.id]={{scores:{{}},best:"",note:""}});

  let descs="";
  order.forEach((sys,k)=>{{
    const L=letters[k];
    descs+=`<div class="desc"><header><span class="letter">${{L}}</span>
      <span style="font-size:.82rem;color:var(--ink-3)">Description ${{L}}</span></header>
      <div class="body">${{esc(it.answers[sys]||"(no answer)")}}</div>
      <div class="axes">`;
    DATA.axes.forEach(([key,name,q,d2,d1,d0])=>{{
      const cur=a.scores?.[L]?.[key];
      descs+=`<div class="ax"><div class="lab"><b>${{name}}</b> — ${{esc(q)}}
        <span>2 = ${{esc(d2)}} · 1 = ${{esc(d1)}} · 0 = ${{esc(d0)}}</span></div>
        <div class="opts">`+
        DATA.scale.map(([v,lbl])=>{{
          const id=`s_${{it.id}}_${{L}}_${{key}}_${{v}}`;
          return `<input type="radio" name="${{it.id}}_${{L}}_${{key}}" id="${{id}}" value="${{v}}"
            ${{cur===v?"checked":""}} data-l="${{L}}" data-k="${{key}}" title="${{esc(lbl)}}">
            <label for="${{id}}" title="${{esc(lbl)}}">${{v}}</label>`;
        }}).join("")+`</div></div>`;
    }});
    descs+=`</div></div>`;
  }});

  $("#app").innerHTML=`
    <div class="split">
      <div class="stack">
        <img class="pic" src="${{it.image}}" alt="Painting ${{S.idx+1}}">
        <div class="card pad">${{refHTML(it.reference)}}</div>
      </div>
      <div>
        <div class="stack" style="margin-bottom:16px">
          <div class="eyebrow">Painting ${{S.idx+1}} of ${{DATA.items.length}}</div>
          <p style="font-size:.88rem">Score each description against the catalogue
          information on the left. Judge what is <b>claimed</b>, not how well it reads.</p>
        </div>
        ${{descs}}
        <div class="choice">
          <div><b>Overall, which description would you rather be given as a gallery visitor?</b>
          <div style="font-size:.82rem;color:var(--ink-3)">Weigh being right above being fluent.</div></div>
          <div class="opts">`+letters.map(L=>{{
            const id=`best_${{it.id}}_${{L}}`;
            return `<input type="radio" name="best_${{it.id}}" id="${{id}}" value="${{L}}" ${{a.best===L?"checked":""}} data-best="1">
              <label for="${{id}}">${{L}}</label>`;
          }}).join("")+`</div>
          <label class="fld" style="margin-top:4px">Anything worth noting? (optional)
            <textarea id="note" placeholder="e.g. B invented a painting that does not exist">${{esc(a.note||"")}}</textarea>
          </label>
        </div>
        <div class="row" style="margin-top:18px;justify-content:space-between">
          <button class="ghost" id="prev" ${{S.idx===0?"disabled":""}}>&larr; Previous</button>
          <span class="legend"><b>${{nDone()}}/${{DATA.items.length}}</b> complete</span>
          <button id="next">${{S.idx===DATA.items.length-1?"Review &amp; submit":"Next &rarr;"}}</button>
        </div>
      </div>
    </div>`;

  $("#app").querySelectorAll(".opts input").forEach(el=>{{
    el.addEventListener("change",()=>{{
      const rec=S.answers[it.id];
      if(el.dataset.best){{ rec.best=el.value; }}
      else{{
        rec.scores[el.dataset.l]=rec.scores[el.dataset.l]||{{}};
        rec.scores[el.dataset.l][el.dataset.k]=Number(el.value);
      }}
      rec.order=Object.fromEntries(letters.map((L,i)=>[L,order[i]]));
      save(); updateBar();
      $("#app").querySelector(".legend b").textContent=`${{nDone()}}/${{DATA.items.length}}`;
    }});
  }});
  $("#note").addEventListener("input",e=>{{ S.answers[it.id].note=e.target.value; save(); }});
  $("#prev").addEventListener("click",()=>{{ if(S.idx>0){{S.idx--;save();render();window.scrollTo(0,0);}} }});
  $("#next").addEventListener("click",()=>{{
    if(S.idx<DATA.items.length-1){{ S.idx++; save(); render(); window.scrollTo(0,0); }}
    else finish();
  }});
  updateBar();
}}

function updateBar(){{
  $("#bar").classList.remove("hide");
  $("#who").textContent=`${{S.code.toUpperCase()}} · ${{S.role==="expert"?"art expert":"general"}}`;
  $("#testflag").classList.toggle("hide",S.mode!=="test");
  const pct=100*nDone()/DATA.items.length;
  $("#pbar").style.width=pct+"%";
  $("#pnum").textContent=`${{nDone()}}/${{DATA.items.length}}`;
}}

// A ~200-character transcript of every judgement, in painting order:
//   ITEM:aaaa,bbbb,cccc,BEST
// where each group is the four axis scores for descriptions A, B and C. The
// letter-to-system mapping is NOT included because it is reproducible from the
// rater code alone (see orderFor); collect_ratings.py rebuilds it.
function shortCode(){{
  const letters=["A","B","C"].slice(0,DATA.systems.length);
  const body=DATA.items.map(it=>{{
    const a=S.answers[it.id]||{{}};
    const groups=letters.map(L=>DATA.axes.map(ax=>{{
      const v=a.scores&&a.scores[L]?a.scores[L][ax[0]]:undefined;
      return (v===undefined||v===null)?"-":String(v);
    }}).join("")).join(",");
    return `${{it.id}}:${{groups}},${{a.best||"-"}}`;
  }}).join(";");
  return `ARTSEEKv3|${{S.code.toUpperCase()}}|${{S.role}}|${{S.mode}}|`+
         `${{new Date().toISOString().slice(0,19)}}Z|${{body}}`;
}}

function payload(){{
  return {{
    rater_code:S.code.toUpperCase(), role:S.role, mode:S.mode,
    prompt_version:"v3-structured", rubric:"0-2 correctness + forced choice",
    started_utc:S.started, submitted_utc:new Date().toISOString(),
    n_items:DATA.items.length, n_complete:nDone(),
    ratings:DATA.items.map(it=>({{
      id:it.id, tier:it.tier, complete:itemDone(it),
      ...(S.answers[it.id]||{{}})
    }}))
  }};
}}

async function finish(){{
  const gaps=DATA.items.map((it,i)=>({{i,it,g:missingFor(it)}})).filter(x=>x.g.length);
  const miss=gaps.length;
  $("#app").classList.add("hide"); $("#finish").classList.remove("hide");
  const j=JSON.stringify(payload(),null,2);
  const stamp=new Date().toISOString().slice(0,10);
  const fname=`artseek-ratings-${{S.mode==="test"?"TEST-":""}}${{S.code.toUpperCase()}}-${{stamp}}.json`;
  const gapList = miss ? `<div class="card pad stack" style="gap:10px">`+
    gaps.map(x=>`<div class="row" style="justify-content:space-between;gap:14px">
      <div><b>Painting ${{x.i+1}}</b> — ${{esc(x.it.reference.title||x.it.id)}}
      <div style="font-size:.8rem;color:var(--ink-3)">${{x.g.map(esc).join(" · ")}}</div></div>
      <button class="ghost jump" data-i="${{x.i}}">Go there</button></div>`).join("")+
    `</div>` : "";
  $("#finish").innerHTML=`
    <div class="eyebrow">Almost done</div>
    <h1>${{miss?`${{miss}} painting${{miss>1?"s":""}} still incomplete`:"All 15 rated — thank you"}}</h1>
    <p>${{miss?"Here is exactly what is missing. You can fill the gaps, or send what you have.":"Save the file below and send it back."}}</p>
    ${{gapList}}
    ${{S.mode==="test"?`<div class="testbanner" style="align-self:flex-start">PRACTICE SESSION — this file is marked as a test</div>`:""}}
    <div class="done stack" style="gap:12px">
      <div><b>Step 1 — copy your answers</b>
      <div style="font-size:.83rem;color:var(--ink-3)">Short code. This is all we need.</div></div>
      <textarea id="short" readonly style="min-height:74px" class="mono">${{esc(shortCode())}}</textarea>
      <div class="row"><button id="cp1">Copy short code</button>
        <span id="status" style="font-size:.85rem"></span></div>
      <div style="margin-top:6px"><b>Step 2 — send it back</b>
      <div style="font-size:.83rem;color:var(--ink-3)">Paste it into an email or message to Nicola.</div></div>
      <details style="margin-top:4px"><summary style="cursor:pointer;font-size:.85rem;color:var(--ink-3)">
        Full record, including your notes (optional)</summary>
        <textarea id="full" readonly style="min-height:150px;margin-top:8px" class="mono">${{esc(j)}}</textarea>
        <div class="row" style="margin-top:8px"><button class="ghost" id="cp2">Copy full record</button></div>
      </details>
    </div>
    <div class="row"><button class="ghost" id="back">Back to painting ${{S.idx+1}}</button></div>`;
  $("#back").addEventListener("click",()=>{{ $("#finish").classList.add("hide"); $("#app").classList.remove("hide"); }});
  $("#finish").querySelectorAll(".jump").forEach(b=>b.addEventListener("click",()=>{{
    S.idx=Number(b.dataset.i); save();
    $("#finish").classList.add("hide"); $("#app").classList.remove("hide");
    render(); window.scrollTo(0,0);
  }}));
  // navigator.clipboard is blocked in many framed contexts, so select-then-
  // execCommand is the fallback, and the textarea stays visible and selectable
  // so a rater can always copy by hand if both fail.
  async function copyFrom(sel,msg){{
    const el=$(sel);
    el.focus(); el.select(); el.setSelectionRange(0,el.value.length);
    try{{ await navigator.clipboard.writeText(el.value); $("#status").textContent=msg; return; }}catch(e){{}}
    try{{
      if(document.execCommand("copy")){{ $("#status").textContent=msg; return; }}
    }}catch(e){{}}
    $("#status").textContent="Select the text above and press Ctrl+C (Cmd+C on a Mac).";
  }}
  $("#cp1").addEventListener("click",()=>copyFrom("#short","Copied. Now paste it into an email or message."));
  $("#cp2").addEventListener("click",()=>copyFrom("#full","Full record copied."));
}}

function validate(){{
  const ok=$("#code").value.trim() && $("#role").value && $("#mode").value;
  $("#start").disabled=!ok;
}}
["#code","#role","#mode"].forEach(s=>$(s).addEventListener("input",validate));
["#role","#mode"].forEach(s=>$(s).addEventListener("change",validate));

$("#start").addEventListener("click",()=>{{
  const code=$("#code").value.trim(), role=$("#role").value, mode=$("#mode").value;
  const prev=load(code,mode);
  if(prev && prev.answers){{ S=prev; S.role=role; }}
  else S={{code,role,mode,idx:0,answers:{{}},started:new Date().toISOString()}};
  S.code=code; S.mode=mode;
  save();
  $("#intro").classList.add("hide"); $("#app").classList.remove("hide");
  render(); window.scrollTo(0,0);
}});
</script>
"""


@click.command()
@click.option("--items-file", default="items.json", show_default=True)
@click.option("--sheets-file", default="factsheets.json", show_default=True)
@click.option("--results-dir", default=str(HERE / "results"), show_default=True)
@click.option("--out-file", default="rating_app.html", show_default=True)
def main(items_file: str, sheets_file: str, results_dir: str, out_file: str):
    items = build_items(items_file, sheets_file, Path(results_dir))
    html = build_html(items)
    out = HERE / out_file
    out.write_text(html, encoding="utf-8")
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({mb:.1f} MB, {len(items)} items x {len(RATED)} systems)")
    missing = [i["id"] for i in items
               if any(not i["answers"][s] for s in RATED)]
    if missing:
        print(f"!! {len(missing)} items missing an answer: {missing}")
    if mb > 15:
        print("!! over 16 MB, shrink the images")


if __name__ == "__main__":
    main()
