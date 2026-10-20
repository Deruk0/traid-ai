"use strict";
// Exercise the actual renderer with a tiny DOM stub, without a browser or network.
const assert=require("node:assert/strict"), fs=require("node:fs"), vm=require("node:vm");
const source=fs.readFileSync("traid_ai/web/app.js","utf8");
const renderer=source.slice(source.indexOf("function eventSearchText(event)"),source.indexOf('const svgNS='));
function node() { return {children:[],textContent:"",classList:{add(){}},append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;}}; }
const scroll={scrollTop:0};
const table=node();table.closest=()=>scroll;
const filter={value:"important"}, search={value:""}, previous={}, next={}, page={}, scope={};
const context={current:null,logSession:null,logOffset:0,logAnchor:null,$:id=>({events:table,"log-filter":filter,"log-search":search,"log-prev":previous,"log-next":next,"log-page":page,"log-scope":scope}[id]),document:{createElement:node},
  time:x=>x,money:x=>String(x),number:x=>String(x),statuses:{},reasons:{}};
vm.createContext(context);vm.runInContext(renderer,context);
function event(kind,payload={}) {return {kind,payload,occurred_at:"time"};}
const fill=event("PAPER_FILL",{side:"BUY",quantity:"1",price:"100",fee:"0"});
const request=event("AI_REQUEST",{snapshot:{snapshot_id:"123456789012"}});
const answer=event("AI_DECISION",{accepted:true,action:"WAIT",reason:"wait"});
const snapshot=event("SNAPSHOT",{bid:"100",ask:"101"});
context.current={events:[request,answer,snapshot,fill]};
context.renderEvents();assert.equal(table.children.length,1);assert.equal(table.children[0].children[1].children[0].textContent,"Покупка");
filter.value="all";context.renderEvents();assert.equal(table.children.length,4);
assert(table.children.some(row=>row.children[3].textContent.includes("Подготовленный снимок")));
filter.value="important";context.current={events:[request,answer,snapshot],important_events:[fill]};
context.renderEvents();assert.equal(table.children.length,1);assert.equal(table.children[0].children[1].children[0].textContent,"Покупка");
context.current={events:[request],important_events:[]};context.renderEvents();assert.equal(table.children[0].children[0].colSpan,5);
console.log("Журнал: фильтр, все события, старые сделки и пустой список — OK");

// Pagination holds the first visible event steady when new events arrive.
filter.value="all";
context.current={selected_id:"a",events:Array.from({length:70},(_,seq)=>({...event("DECISION",{reason:"NO_SIGNAL"}),seq}))};
context.renderEvents();assert.equal(table.children.length,25);assert.equal(next.disabled,false);
context.logOffset=25;context.logAnchor=null;context.renderEvents();assert.equal(context.logAnchor,44);
context.current.events.push({...event("DECISION",{reason:"NO_SIGNAL"}),seq:70});
context.renderEvents();assert.equal(context.logAnchor,44);assert.equal(context.logOffset,26);
search.value="покупка";context.current.events.push({...fill,seq:71});context.resetLog();
assert.equal(table.children.length,1);assert.equal(next.disabled,true);assert.equal(previous.disabled,true);
search.value="does-not-exist";context.resetLog();assert(table.children[0].children[0].textContent.includes("запросу"));
search.value="";context.current.selected_id="b";context.renderEvents();assert.equal(context.logOffset,0);
console.log("Журнал: поиск, страницы, устойчивость при обновлении и смена сессии — OK");

scroll.scrollTop=120;context.renderEvents();assert.equal(scroll.scrollTop,120);context.resetLog();assert.equal(scroll.scrollTop,0);

// Dedicated blockers are supplied from full history, independent of recent events.
filter.value="blockers";
const blocker={seq:"9:test",kind:"TRADE_BLOCKER",occurred_at:"time",payload:{scope:"Выход",component:"Брокер / сверка",code:"BROKER_NOT_READY",detail:"Ждём сверку",status:"Блокировка",source_kind:"RISK_VERDICT",source_seq:9,count:3,first_at:"start",facts:{actual_cash:"100"}}};
context.current={selected_id:"b",capabilities:["trade-blocking-log-v1"],events:[snapshot],blocking_events:[blocker]};
context.resetLog();assert.equal(table.children.length,1);
assert.equal(table.children[0].children[1].children[0].textContent,"Выход");
assert(table.children[0].children[3].textContent.includes("повторов: 3"));
assert.equal(table.children[0].children[4].textContent,"Блокировка");
search.value="брокер";context.resetLog();assert.equal(table.children.length,1);
search.value="";context.current={events:[]};context.resetLog();assert(scope.textContent.includes("не поддерживает"));
console.log("Журнал: отдельный список блокировок, источник, поиск и повторы — OK");

// The source has its own column; missing metadata never invents a blocker.
filter.value="blockers";search.value="";
for (const component of ["Qwen","Фильтры кандидата","Risk Engine","Брокер / сверка"]) {
  context.current={blocking_events:[{...blocker,payload:{...blocker.payload,component}}],capabilities:["trade-blocking-log-v1"]};
  context.resetLog();assert.equal(table.children[0].children.length,5);
  assert.equal(table.children[0].children[2].textContent,component);
}
context.current.blocking_events[0].payload.component=null;context.renderEvents();
assert.equal(table.children[0].children[2].textContent,"Не указан");
filter.value="all";
context.current={events:[{...event("RISK_VERDICT",{approved:false,reason:"BROKER_NOT_READY"}),seq:9}],blocking_events:[blocker]};
context.resetLog();assert.equal(table.children[0].children[2].textContent,"Брокер / сверка");
assert.equal(table.children[0].children[4].textContent,"Отказ");
context.current.events[0].seq=10;context.renderEvents();assert.equal(table.children[0].children[2].textContent,"—");
console.log("Столбец источника: отдельная ячейка, связь с исходным событием и отсутствие догадок — OK");
