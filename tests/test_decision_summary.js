"use strict";
const assert=require("node:assert/strict"), fs=require("node:fs"), vm=require("node:vm");
const source=fs.readFileSync("traid_ai/web/app.js","utf8");
const renderer=source.slice(source.indexOf("function renderDecisionSummary(data)"),source.indexOf("modeHelp(); refresh();"));
function node() { return {textContent:"",hidden:false,children:[],replaceChildren(){this.children=[];},append(item){this.children.push(item);}}; }
const nodes=new Map();
const context={$:id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);},document:{createElement:node}};
vm.createContext(context);vm.runInContext(renderer,context);
context.renderDecisionSummary({decision_summary:{requests:82,requests_with_entry_option:36,accepted_entries:5,
  accepted_waits:39,rejected:38,unanswered:0,blocked:46,blocking_conditions:[],warnings:[],
  risk_denials:5,fills:0,risk_reason_counts:{BROKER_NOT_READY:5},
  rejected_reason_counts:{"Снимок устарел или портфель изменился":19}}});
assert(nodes.get("decision-summary-counts").textContent.includes("предложений входа: 5"));
assert(nodes.get("decision-summary-blocks").children.some(n=>n.textContent.includes("Брокер не готов") && n.textContent.includes("5")));
assert(nodes.get("decision-summary-blocks").children.some(n=>n.textContent.includes("Снимок устарел") && n.textContent.includes("19")));
assert(nodes.get("decision-summary-note").textContent.includes("Предложение ИИ не означает отправку ордера"));
console.log("Диагностика: предложение входа, отказ брокера и отклонённый ответ различаются — OK");
