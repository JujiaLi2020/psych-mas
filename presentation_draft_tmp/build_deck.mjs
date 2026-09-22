import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "C:/research/2026-2-psych-mas/PsyMAS_AIME_Presentation_Draft.pptx";
const PREVIEW = "C:/research/2026-2-psych-mas/presentation_draft_tmp";
const W = 1280, H = 720;
const C = {
  navy: "#102A43", ink: "#16324F", muted: "#61758A", teal: "#176B78",
  cyan: "#25A7C9", orange: "#C98213", red: "#C93232", purple: "#5B4AA5",
  green: "#278B67", bg: "#F4F7FA", white: "#FFFFFF", line: "#D5DEE8",
  paleBlue: "#EAF3F8", paleOrange: "#FFF5E3", paleRed: "#FFF0F0", paleGray: "#E9EFF5",
};

function addText(slide, text, x, y, w, h, opts={}) {
  const s = slide.shapes.add({ geometry: "textbox", position: {left:x, top:y, width:w, height:h}, fill:"none", line:{style:"solid", fill:"none", width:0} });
  s.text = text;
  s.text.style = { fontSize: opts.size || 20, color: opts.color || C.ink, bold: !!opts.bold, italic: !!opts.italic, alignment: opts.align || "left" };
  return s;
}
function box(slide, x, y, w, h, fill=C.white, line=C.line, radius="rounded-lg") {
  return slide.shapes.add({ geometry:"roundRect", position:{left:x,top:y,width:w,height:h}, fill, line:{style:"solid",fill:line,width:1}, borderRadius:radius });
}
function rule(slide, x, y, w, color=C.line, height=2) {
  return slide.shapes.add({ geometry:"rect", position:{left:x,top:y,width:w,height:height}, fill:color, line:{style:"solid",fill:color,width:0} });
}
function circle(slide, x, y, d, fill, text, textColor=C.white) {
  slide.shapes.add({ geometry:"ellipse", position:{left:x,top:y,width:d,height:d}, fill, line:{style:"solid",fill, width:0} });
  addText(slide, text, x, y+4, d, d-8, {size:18,bold:true,color:textColor,align:"center"});
}
function section(slide, num, title, kicker) {
  circle(slide, 58, 42, 42, C.teal, String(num));
  addText(slide, kicker.toUpperCase(), 116, 40, 250, 20, {size:14,bold:true,color:C.teal});
  addText(slide, title, 116, 64, 1080, 52, {size:34,bold:true,color:C.navy});
  rule(slide, 116, 124, 1048, C.teal, 3);
}
function footer(slide, text="PsyMAS · AIME Con 2026") { addText(slide,text,58,680,600,18,{size:12,color:C.muted}); }
function note(slide, lines) { slide.speakerNotes.textFrame.setText(["[Sources]", ...lines]); }

function newSlide(p, fill=C.bg) { const s=p.slides.add(); s.background.fill=fill; return s; }

async function main() {
  const p = Presentation.create({ slideSize:{width:W,height:H} });

  // 1
  { const s=newSlide(p,C.navy); s.background.fill=C.navy;
    addText(s,"PsyMAS",72,92,600,76,{size:64,bold:true,color:C.white});
    addText(s,"Test Security Analytics",76,174,660,48,{size:34,bold:true,color:"#8FE3E7"});
    addText(s,"An evidence-governed human-in-the-loop workbench for psychometric forensics",76,272,760,90,{size:26,color:C.white});
    box(s,76,430,530,100,"#173B55","#2B8190");
    addText(s,"Statistical flags are review triggers — not misconduct conclusions.",102,458,480,50,{size:22,bold:true,color:C.white});
    addText(s,"AIME Con 2026 · Work in progress presentation",76,634,620,24,{size:16,color:"#B9CFDD"});
    // visual mark
    for(let i=0;i<5;i++){ rule(s,850+i*48,430-i*28,34,[C.cyan,C.teal,C.orange,C.red,C.purple][i],18); }
    note(s,["PsyMAS project materials and user-provided prototype description.","No external visual assets used."]); footer(s,"PsyMAS · AIME Con 2026");
  }

  // 2
  { const s=newSlide(p); section(s,1,"Test-security evidence is easy to generate — and easy to over-interpret","The problem");
    addText(s,"A detector can return many statistics. A reviewer still needs to know what counts, why it matters, and what remains uncertain.",116,158,1000,48,{size:24,color:C.ink});
    const items=[
      ["Many outputs","One run can produce flags, p-values, thresholds, plots, and auxiliary summaries.",C.cyan],
      ["Unequal evidentiary weight","Package flags, calibrated rules, and display-only diagnostics cannot be treated alike.",C.orange],
      ["Human decision remains necessary","A flagged pattern may reflect disengagement, context, data limitations, or something more serious.",C.red],
    ];
    items.forEach((it,i)=>{ const x=116+i*350; box(s,x,278,310,194,C.white,it[2]); addText(s,it[0],x+22,305,270,32,{size:23,bold:true,color:C.navy}); addText(s,it[1],x+22,354,260,85,{size:18,color:C.muted}); rule(s,x+22,444,65,it[2],5); });
    addText(s,"Design question",116,548,220,26,{size:18,bold:true,color:C.teal});
    addText(s,"How can statistical outputs become a traceable review record without becoming an automated verdict?",116,578,950,45,{size:26,bold:true,color:C.navy}); footer(s); note(s,["Mislevy (1994), evidence-centered assessment framing.","PsyMAS project materials."]); }

  // 3
  { const s=newSlide(p); section(s,2,"PsyMAS separates computation, governance, explanation, and adjudication","Design principles");
    const cols=[
      ["01","Compute","Deterministic psychometric and forensic routines produce the raw outputs.",C.cyan],
      ["02","Govern","An index registry and rulebook decide what may count as evidence.",C.orange],
      ["03","Explain","A constrained LLM summarizes governed evidence and selected raw-data context.",C.purple],
      ["04","Review","A human reviewer confirms the final procedural outcome.",C.red],
    ];
    cols.forEach((it,i)=>{ const x=116+i*260; circle(s,x,220,54,it[3],it[0]); addText(s,it[1],x,300,220,32,{size:23,bold:true,color:C.navy}); addText(s,it[2],x,350,210,120,{size:18,color:C.muted}); if(i<3) addText(s,"→",x+218,232,34,34,{size:28,bold:true,color:C.teal,align:"center"}); });
    box(s,116,530,1000,70,C.paleBlue,C.teal); addText(s,"Core boundary: the system organizes evidence for review; it does not infer intent or determine misconduct.",140,550,950,28,{size:22,bold:true,color:C.navy}); footer(s); note(s,["Amershi et al. (2019), human-centered AI principles.","PsyMAS project materials."]); }

  // 4
  { const s=newSlide(p); section(s,3,"The workflow moves from assessment data to a traceable human decision","Workflow");
    const labels=["Assessment\ndata","Deterministic\nevidence","Evidence\ngovernance","AI-assisted\nreview","Human\nreview","Review\nrecord"];
    labels.forEach((lab,i)=>{ const x=90+i*190; box(s,x,260,156,116,i===3?C.paleBlue:C.white,i===3?C.teal:C.line); circle(s,x+52,206,50,[C.cyan,C.cyan,C.orange,C.purple,C.red,C.green][i],String(i+1)); addText(s,lab,x+12,294,132,50,{size:20,bold:true,color:C.navy,align:"center"}); if(i<5) addText(s,"→",x+158,302,32,28,{size:26,bold:true,color:C.teal,align:"center"}); });
    addText(s,"At each stage, PsyMAS preserves the link between source data, index output, rule, domain profile, and review language.",116,470,1020,38,{size:23,color:C.ink});
    addText(s,"The result is not a single score. It is a reviewable chain of evidence.",116,540,960,44,{size:28,bold:true,color:C.navy}); footer(s); note(s,["PsyMAS project workflow and rulebook design."]); }

  // 5
  { const s=newSlide(p); section(s,4,"The same case can combine performance, timing, exposure, and answer-change records","Assessment inputs");
    const data=[
      ["Responses","Item-level 0/1 scores","Baseline psychometric evidence"],
      ["Response times","Item-level timing","Rapid-guessing and timing context"],
      ["Item / exposure metadata","Parameters and compromised items","Preknowledge review"],
      ["Answer-change records","Initial and final responses","Tampering / answer-change review"],
    ];
    data.forEach((r,i)=>{const y=190+i*92; rule(s,116,y+8,10,[C.cyan,C.orange,C.purple,C.red][i],58); addText(s,r[0],150,y,250,30,{size:22,bold:true,color:C.navy}); addText(s,r[1],440,y,290,30,{size:20,color:C.ink}); addText(s,r[2],790,y,380,30,{size:19,color:C.muted});});
    box(s,116,590,1000,42,C.paleOrange,C.orange); addText(s,"Optional inputs expand the review context; missing inputs remain visible as missing evidence.",138,601,950,22,{size:18,bold:true,color:C.navy}); footer(s); note(s,["PsyMAS semi-simulated data design and current input checklist."]); }

  // 6
  { const s=newSlide(p); section(s,5,"Indices are assigned to the behavior they are intended to examine","From indices to domains");
    const left=["rg:CUMP","rg:NT","pk:L_T","pk:W_T","tt:EDI_SD family","tt:GBT_SD","pm:person-fit family"];
    const domains=["Response-Time","Preknowledge","Tampering","Misfit\n(supporting)"];
    left.forEach((v,i)=>{const y=172+i*52; box(s,116,y,210,34,C.paleBlue,C.cyan); addText(s,v,130,y+7,180,20,{size:17,bold:true,color:C.navy});});
    domains.forEach((v,i)=>{const y=190+i*90; box(s,550,y,260,56,i===3?C.paleGray:C.white,[C.orange,C.purple,C.red,C.muted][i]); addText(s,v,570,y+15,220,28,{size:20,bold:true,color:C.navy,align:"center"});});
    addText(s,"→",380,326,90,34,{size:30,bold:true,color:C.teal,align:"center"}); addText(s,"One index family → one governed path",372,398,250,42,{size:18,bold:true,color:C.teal,align:"center"});
    box(s,930,250,190,100,C.paleRed,C.red); addText(s,"Review priority",950,274,150,25,{size:20,bold:true,color:C.navy,align:"center"}); addText(s,"only after\ndomain synthesis",950,308,150,35,{size:17,color:C.muted,align:"center"});
    addText(s,"Correction variants are collapsed before domain strength is calculated; display-only outputs remain visible but do not increase priority.",116,590,1020,36,{size:20,color:C.ink}); footer(s); note(s,["Gorney & Deng (2024), aberrance package.","PsyMAS Index Registry and evidence aggregation rules."]); }

  // 7
  { const s=newSlide(p); section(s,6,"A case-level lineage makes the priority recommendation inspectable","Worked example");
    addText(s,"Examinee 332",116,162,250,34,{size:24,bold:true,color:C.navy}); addText(s,"4 index families · 2 priority domains · supporting misfit context",116,198,500,24,{size:18,color:C.muted});
    const stages=["Index families","Evidence domains","Rules","Review priority"];
    stages.forEach((t,i)=>addText(s,t,140+i*265,252,210,22,{size:18,bold:true,color:C.teal,align:"center"}));
    const rows=[ ["rg:CUMP / NT","Response-Time","RT-B3-04","Moderate"],["tt:EDI_SD / GBT_SD","Tampering","TP-B3-05","Strong"],["pm families","Misfit (supporting)","MF-B3-01","Not counted"] ];
    rows.forEach((r,ri)=>{const y=310+ri*85; r.forEach((v,i)=>{const x=120+i*265; const fill=i===3?(ri===1?C.red:ri===0?C.orange:C.paleGray):C.white; box(s,x,y,230,48,fill,i===3?fill:C.line); addText(s,v,x+10,y+13,210,24,{size:17,bold:i===3,color:i===3?C.white:C.navy,align:"center"});});});
    addText(s,"→",348,322,35,25,{size:23,bold:true,color:C.teal,align:"center"}); addText(s,"→",613,322,35,25,{size:23,bold:true,color:C.teal,align:"center"}); addText(s,"→",878,322,35,25,{size:23,bold:true,color:C.teal,align:"center"});
    box(s,116,600,1000,44,C.paleBlue,C.teal); addText(s,"Critical / Expedited: a workflow priority for human review, not a misconduct finding.",138,612,960,24,{size:20,bold:true,color:C.navy}); footer(s); note(s,["Example values reflect the current PsyMAS worked-example lineage for Examinee 332.","PsyMAS rulebook and evidence store."]); }

  // 8
  { const s=newSlide(p); section(s,7,"The reviewer sees context before interpreting a flag","Case context");
    const panels=[
      ["Score percentile","85th percentile","Where the examinee sits in the cohort",C.paleBlue],
      ["Item accuracy","Correct / incorrect by item","Which responses differ from item context",C.paleOrange],
      ["Score vs effort","Ability × response-time effort","Whether high performance and low effort co-occur",C.paleRed],
      ["Response-time profile","Examinee vs item mean","Where timing departs from expected timing",C.paleBlue],
    ];
    panels.forEach((pnl,i)=>{const x=116+(i%2)*520, y=180+Math.floor(i/2)*180; box(s,x,y,470,142,C.white,C.line); rule(s,x,y,8,pnl[3],142); addText(s,pnl[0],x+28,y+22,400,28,{size:22,bold:true,color:C.navy}); addText(s,pnl[1],x+28,y+63,400,24,{size:19,bold:true,color:C.teal}); addText(s,pnl[2],x+28,y+100,400,24,{size:17,color:C.muted});});
    addText(s,"These plots clarify the flagged pattern; they do not replace the governed evidence record.",116,580,1000,34,{size:22,bold:true,color:C.navy}); footer(s); note(s,["PsyMAS Examinee Report views and project screenshots supplied in the working context."]); }

  // 9
  { const s=newSlide(p); section(s,8,"Preknowledge and change-point views add targeted context — not extra verdicts","Contextual views");
    box(s,116,180,500,310,C.white,C.purple); addText(s,"Preknowledge view",142,208,400,30,{size:23,bold:true,color:C.navy}); addText(s,"Compare selected performance and timing on exposed items with cohort means.",142,250,410,55,{size:19,color:C.muted});
    for(let i=0;i<7;i++){ const x=160+i*55; const h=[70,110,90,145,100,130,85][i]; rule(s,x,420-h,28,h,i%3===0?C.red:C.green,); rule(s,x,420,28,3,C.navy); } addText(s,"item number",285,448,160,24,{size:16,color:C.muted,align:"center"});
    box(s,660,180,456,310,C.white,C.purple); addText(s,"Change-point view",686,208,400,30,{size:23,bold:true,color:C.navy}); addText(s,"Locate where multiple methods estimate a shift in score or response-time behavior.",686,250,390,55,{size:19,color:C.muted}); rule(s,708,390,350,4,C.line); [760,850,1008].forEach((x,i)=>circle(s,x,370,28,[C.orange,C.purple,C.red][i],"")); rule(s,930,330,4,120,C.teal); addText(s,"estimated shift zone",875,430,180,22,{size:16,bold:true,color:C.teal,align:"center"});
    box(s,116,560,1000,44,C.paleGray,C.line); addText(s,"Use these views to decide what to inspect next; change-point output is localization-only unless calibrated for evidence entry.",138,572,960,22,{size:18,color:C.ink}); footer(s); note(s,["PsyMAS preknowledge and change-pattern contextual views."]); }

  // 10
  { const s=newSlide(p); section(s,9,"The LLM explains governed evidence within a hard boundary","AI-assisted review");
    box(s,116,180,520,350,C.white,C.teal); addText(s,"Evidence-grounded prompt",142,208,430,30,{size:23,bold:true,color:C.navy}); const bullets=["Use eligible index families and domain profiles.","Use selected raw-data summaries to clarify the pattern.","Explain what the reviewer should inspect next.","Disclose unavailable or calibration-limited evidence."]; bullets.forEach((b,i)=>{circle(s,144,268+i*58,24,C.teal,String(i+1)); addText(s,b,184,269+i*58,410,28,{size:18,color:C.ink});});
    box(s,700,180,416,350,C.navy,C.navy); addText(s,"Never allowed",730,208,350,30,{size:23,bold:true,color:C.white}); ["Compute indices","Create or alter flags","Change thresholds","Infer intent","Determine misconduct"].forEach((b,i)=>{ addText(s,"×",732,270+i*44,24,26,{size:23,bold:true,color:"#FF8A8A"}); addText(s,b,770,272+i*44,280,24,{size:18,color:C.white}); });
    addText(s,"The LLM drafts language. The reviewer owns the decision.",116,590,1000,32,{size:24,bold:true,color:C.navy}); footer(s); note(s,["Amershi et al. (2019), human-centered AI principles.","PsyMAS prompt constraints and LLM configuration."]); }

  // 11
  { const s=newSlide(p); section(s,10,"Human review converts evidence into a documented procedural outcome","Human review and audit");
    const steps=[["Review","Inspect the lineage, plots, and source records."],["Question","Ask the LLM for clarification without changing the record."],["Confirm","Select a procedural outcome and write a reviewer note."],["Audit","Preserve the decision, evidence links, and run identifier."]];
    steps.forEach((st,i)=>{const x=116+i*260; circle(s,x+72,190,52,[C.cyan,C.purple,C.orange,C.green][i],String(i+1)); addText(s,st[0],x,270,200,30,{size:23,bold:true,color:C.navy,align:"center"}); addText(s,st[1],x,318,210,80,{size:18,color:C.muted,align:"center"}); if(i<3) addText(s,"→",x+215,204,36,30,{size:28,bold:true,color:C.teal,align:"center"});});
    box(s,116,500,1000,90,C.paleBlue,C.teal); addText(s,"A flag is a review trigger. The final record distinguishes system evidence from human adjudication.",142,530,950,30,{size:23,bold:true,color:C.navy}); footer(s); note(s,["PsyMAS human-review record and audit controls."]); }

  // 12
  { const s=newSlide(p,C.navy); addText(s,"What PsyMAS contributes",72,72,900,64,{size:50,bold:true,color:C.white});
    const take=["Traceability from index to review priority","Conservative handling of thresholds and missing evidence","Plain-language assistance without automated culpability","A reproducible workbench for worked examples and future evaluation"];
    take.forEach((t,i)=>{circle(s,90,194+i*72,34,[C.cyan,C.orange,C.purple,C.green][i],"✓"); addText(s,t,146,196+i*72,920,32,{size:24,bold:true,color:C.white});});
    box(s,72,550,970,70,"#173B55","#2B8190"); addText(s,"The next step is validation with controlled scenarios and expert review — not replacing the reviewer.",98,570,920,28,{size:21,bold:true,color:C.white}); addText(s,"PsyMAS · Test Security Analytics",72,660,500,20,{size:15,color:"#B9CFDD"}); note(s,["PsyMAS project materials and current prototype scope."]); }

  await fs.mkdir(PREVIEW,{recursive:true});
  for (const [i,slide] of p.slides.items.entries()) {
    const stem=`slide-${String(i+1).padStart(2,"0")}`;
    const png=await p.export({slide,format:"png",scale:1});
    await fs.writeFile(`${PREVIEW}/${stem}.png`,new Uint8Array(await png.arrayBuffer()));
    const layout=await slide.export({format:"layout"});
    await fs.writeFile(`${PREVIEW}/${stem}.layout.json`,await layout.text());
  }
  const pptx=await PresentationFile.exportPptx(p); await pptx.save(OUT);
}
main().catch(e=>{console.error(e);process.exitCode=1;});
