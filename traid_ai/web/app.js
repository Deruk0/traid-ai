"use strict";
const $ = id => document.getElementById(id);
const number = (value, digits = 2) => Number(value).toLocaleString("ru-RU", {minimumFractionDigits: digits, maximumFractionDigits: digits});
const money = value => `${number(value)} $`;
const time = value => value ? new Date(value).toLocaleTimeString("ru-RU", {hour:"2-digit", minute:"2-digit", second:"2-digit"}) : "—";
const statuses = {RUNNING:"Сессия работает", PAUSED_ENTRIES:"Входы на паузе", FLATTENING:"Закрываем позиции", UNRESOLVED:"Закрытие или сверка не подтверждены", CLOSED:"Сессия завершена"};
const reasons = {WARMUP:"Собираем первые наблюдения", NO_SIGNAL:"Подходящего сигнала нет", MOMENTUM_ABOVE_MEAN:"Цена выше среднего: тестовый сигнал", MOMENTUM_BELOW_MEAN:"Цена ниже среднего: тестовый выход", STOP_LOSS:"Защитный выход", MAX_HOLD:"Достигнут максимум удержания", DEADLINE:"Завершение по времени", USER_STOP:"Завершение по твоей команде", SESSION_LOSS_LIMIT:"Достигнут лимит потери", DRAWDOWN_LIMIT:"Достигнут лимит просадки", WITHIN_LIMITS:"Размер сделки в пределах лимитов", BUDGET_OR_MINIMUM:"Недостаточный бюджет или размер сделки", DECISION_UNAVAILABLE:"Модуль решений недоступен: ждём", ENTRIES_PAUSED:"Новые входы запрещены паузой", RECOVERY:"Выход после восстановления"};
const sourceNames = {synthetic:"Демо", "kraken-public":"Локальный рынок", "alpaca-paper":"Alpaca Paper"};
Object.assign(reasons, {EXECUTION_TEST:"Проверка исполнения", EXECUTION_TEST_COMPLETE:"Проверка завершена", STRATEGY_ENTRY:"Тестовый сигнал", BROKER_RECONCILIATION:"Не подтверждена сверка с Alpaca", BROKER_NOT_READY:"Исполнитель пока не готов", ORDER_REJECTED:"Alpaca отклонила заявку", ORDER_NOT_ACCEPTED:"Заявка отсутствует после сверки", NO_ORDER_OR_ACCOUNT_CHANGE:"Заявка не найдена; баланс и позиция проверены", QUOTE_EXPIRED_BEFORE_SUBMIT:"Котировка устарела до отправки"});
Object.assign(reasons, {QWEN_DECISION:"Решение Qwen",AI_PENDING:"Qwen анализирует снимок",AI_COOLDOWN:"Пауза между запросами Qwen",AI_NOT_NEEDED:"Сейчас управление выполняет код"});
Object.assign(reasons,{REDUCE_ONLY:"Разрешено уменьшение позиции",WORKER_STALLED:"Исполнитель не подтверждает прогресс",SYSTEM_RESUME_GAP:"Обнаружен длительный перерыв",WORKER_ERROR:"Ошибка исполнителя"});
Object.assign(reasons,{AI_REVIEW_SCHEDULED:"Ждём конкретного события или срока пересмотра"});
Object.assign(reasons,{TRAILING_STOP:"Выход по скользящему стопу",TRAILING_POSITION_HOLD:"Позицию сопровождает скользящий стоп"});
Object.assign(reasons,{BROKER_RECONCILING:"Решение ожидает сверки брокера"});
let current = null, selected = null, token = null, busy = false, fetching = false;
const priceChart = new PriceChart();
let logOffset = 0, logAnchor = null, logSession = null, toastTimer;
let deleteTarget = null;
selected = new URLSearchParams(location.search).get("session");
let modelChoiceInitialized = false;
const modelName = model => ({"qwen3.5:4b":"Qwen 3.5 4B","qwen3.5:9b":"Qwen 3.5 9B"}[model] || model || "Qwen 3.5 4B");

function alertMessage(message) { $("alert").hidden = !message; $("alert").textContent = message || ""; }
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json", "X-Control-Token":token}, body:JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) { const error = new Error(data.error || "Не удалось выполнить запрос"); error.status = response.status; throw error; }
  return data;
}
async function refresh() {
  if (fetching) return;
  fetching = true;
  try {
    const requested = selected;
    const data = await api(`/api/state${selected ? `?session=${encodeURIComponent(selected)}` : ""}`);
    if (requested !== selected) return;
    token = data.control_token; current = data; selected = data.selected_id;
    const url = new URL(location.href);
    if (selected) url.searchParams.set("session", selected); else url.searchParams.delete("session");
    history.replaceState(null, "", url);
    $("connection").textContent = "Панель подключена"; $("connection").classList.remove("offline");
    render(data);
  } catch (error) {
    if (error.status === 400 && error.message === "Сессия не найдена" && selected) { selected = null; toast("Сессия удалена или недоступна. Открываем последнюю."); return; }
    $("connection").textContent = "Нет связи с сервером"; $("connection").classList.add("offline");
    alertMessage(`${error.message}. Проверь, что окно сервера запущено.`);
  } finally { fetching = false; }
}
async function command(path, payload) {
  if (busy) return;
  busy = true; alertMessage(""); renderButtons();
  try { const data = await api(path, payload); selected = data.session_id; await refresh(); }
  catch (error) { alertMessage(error.message); }
  finally { busy = false; renderButtons(); }
}
function renderButtons() {
  const running = current?.running || false;
  const active = running && current.selected_id === current.active_id;
  const closed = current?.report?.status === "CLOSED";
  const checking = current?.maintenance?.running || false;
  $("compare").disabled = busy || running || checking || !closed || !token;
  $("reconcile-fees").disabled = busy || running || checking || !["CLOSED","UNRESOLVED"].includes(current?.report?.status) || !token || current?.config?.source !== "alpaca-paper" || !current?.alpaca_connected;
  $("start").disabled = busy || running || checking || !token || !current?.capabilities?.includes("qwen-model-selection-v2") || ($("mode").value === "alpaca" && !current?.alpaca_connected) || ($("strategy").value === "qwen" && (!current?.model?.connected || current.model.model !== $("llm-model").value));
  $("start").textContent = busy ? "Выполняем команду…" : running ? "Сессия запущена" : "▶ Запустить сессию";
  for (const input of $("settings").querySelectorAll("input,select")) input.disabled = running || busy;
  $("pause").disabled = busy || !active || !["RUNNING", "PAUSED_ENTRIES"].includes(current?.report?.status);
  $("pause").textContent = current?.report?.status === "PAUSED_ENTRIES" ? "Разрешить входы" : "Пауза входов";
  $("stop").disabled = busy || !active;
  $("restore").disabled = busy || running || checking || !current?.report || closed;
  $("export").disabled = !current?.report || busy;
  $("sessions").disabled = busy || !current?.sessions?.length;
  $("copy-session").disabled = !current?.selected_id;
  $("active-session").disabled = busy || !running || active;
  $("delete-session").disabled = busy || running || checking || !closed || !token || !current?.capabilities?.includes("panel-usability-v1") || Number(current?.state?.quantity) !== 0 || current?.state?.broker?.pending != null;
  $("mode").querySelector("option[value=alpaca]").disabled = !current?.alpaca_connected;
  $("strategy").querySelector("option[value=smoke]").disabled = $("mode").value !== "alpaca";
  $("strategy").querySelector("option[value=qwen]").disabled = $("mode").value === "demo";
  $("llm-model").disabled = running || busy || current?.model?.checking;
  $("model-connect").disabled = !token || running || busy || current?.model?.checking;
}
function colorValue(id, value) { $(id).classList.toggle("positive", Number(value) > 0); $(id).classList.toggle("negative", Number(value) < 0); }
function renderModel(data) {
  const model = data.model;
  if (!modelChoiceInitialized) { $("llm-model").value = model?.model || "qwen3.5:4b"; modelChoiceInitialized = true; }
  const chosen = $("llm-model").value;
  $("model-tag").textContent = modelName(chosen);
  $("model-status").textContent = model?.checking ? `Загружаем и проверяем ${modelName(model.model)}…` : model?.connected ? model.model === chosen ? `Подключена ${modelName(model.model)} · ${model.info.quantization} · проверка ${number(model.last_answer.latency_ms / 1000,1)} с` : `Подключена ${modelName(model.model)}. Для ${modelName(chosen)} нажми «Подключить Qwen».` : (model?.model === chosen && model?.error) || "Запусти Ollama и нажми «Подключить Qwen».";
}
function render(data) {
  renderModel(data);
  const latestAnswer = [...data.events].reverse().find(e => e.kind === "AI_DECISION" && e.payload.accepted)?.payload;
  const ai = data.report?.status === "CLOSED" ? latestAnswer || data.report?.ai : data.report?.ai;
  $("ai-panel").hidden = data.config?.strategy_version !== "qwen-local-v1";
  const interrupted=data.health?.recovery_required && ai?.status === "PENDING";
  $("ai-status").textContent = interrupted ? "Запрос прерван" : {PENDING:"Думает",READY:"Ответ проверен",REJECTED:"Ответ не принят",DISCARDED:"Запрос отменён"}[ai?.status] || "Собираем данные";
  $("ai-reason").textContent = ai ? `${{WAIT:"Ждать",REQUEST_ENTRY:"Предлагает вход",REQUEST_EXIT:"Предлагает выход"}[ai.action] || "Ждать"} · ${ai.reason}` : "Собираем наблюдения для расчётов и памяти сессии.";
  if(interrupted) $("ai-reason").textContent="Ответ на последний запрос не сохранён. В этой панели исполнитель сессии не запущен; для продолжения требуется восстановление.";
  $("ai-details").textContent = ai ? `${modelName(ai.model || data.config?.llm_model)} · данные ${time(ai.observed_at)}${ai.latency_ms != null ? ` · ответ ${number(ai.latency_ms/1000,1)} с` : ""} · снимок ${ai.snapshot_id.slice(0,12)}` : "Риск проверяется независимо от ИИ.";
  if(ai?.error_code) $("ai-details").textContent+=` · причина: ${ai.error_code}`;
  if(ai?.result_wait_ms != null) $("ai-details").textContent+=` · обработка после готовности: ${number(ai.result_wait_ms/1000,2)} с`;
  if(ai?.review_at) $("ai-details").textContent+=` · пересмотр: следующая закрытая свеча или ${time(ai.review_at)} · последовательных отсрочек: ${ai.wait_streak || 1}`;
  const evidenceLabels={candle_status:"Свечи",candle_count:"Число свечей",c5_status:"Окно 5 мин",c15_status:"Окно 15 мин",c5_return_pct:"Изменение 5 мин, %",c15_return_pct:"Изменение 15 мин, %",c5_volume_btc:"Объём 5 мин, BTC",c15_range_pct:"Диапазон 15 мин, %",round_trip_cost_pct:"Оценка расходов, %",seconds_to_flatten:"До сворачивания, с",position_qty:"Позиция, BTC",session_status:"Сессия",pending_order:"Заявка в обработке"};
  $("ai-evidence").textContent=ai?.explanation_status === "VERIFIED" ? `Факты причины проверены · ${ai.reason_code} · `+Object.entries(ai.evidence || {}).map(([key,value])=>`${evidenceLabels[key] || key}: ${value == null ? "недоступно" : value === true ? "да" : value === false ? "нет" : value}`).join(" · ") : ai && data.config?.decision_policy === "legacy" ? "Архивная сессия: объяснение по прежним правилам. Проверяемые причины доступны в новой сессии." : "";
  const context = ai?.context_summary;
  const horizons = context?.ready_horizons_seconds?.map(s => `${s / 60} мин`).join(", ");
  const candidateStatus={ELIGIBLE:"обязательные условия выполнены",BLOCKED:"обязательные условия не выполнены",NOT_APPLICABLE:"позиция уже открыта"}[context?.entry_candidate_status];
  $("ai-context").textContent = context ? `${context.sample_count} наблюдений · ${horizons ? "готовые окна: " + horizons : "окна ещё набираются"}${context.gaps ? " · есть пропуски" : ""} · событий в памяти: ${context.memory_event_count}. Свечей: ${context.candle_count || 0} · ${context.candle_status || "UNAVAILABLE"} · ${context.volume_available ? "объёмы получены" : "объёмы недоступны"}.${candidateStatus ? " Кандидат на вход: " + candidateStatus + "." : ""}` : "";
  renderSessions(data);
  const r = data.report, state = data.state;
  const objective=r?.objective;
  $("objective-panel").hidden=!objective;
  if(objective) {
    $("objective-status").textContent=objective.reached === true ? "Достигнута по оценке" : objective.reached === null ? "Нет оценки" : "Необязательная цель";
    $("objective-target").textContent=`${number(objective.target_percent)} % · ${money(objective.target_pnl_usd)}`;
    $("objective-current").textContent=objective.estimated_net_pnl_usd == null ? "Нет подтверждённой котировки" : money(objective.estimated_net_pnl_usd);
    $("objective-remaining").textContent=objective.remaining_usd == null ? "—" : money(objective.remaining_usd);
    $("objective-progress").hidden=objective.progress_percent == null;
    $("objective-progress").value=objective.progress_percent || 0;
  }
  if (r) {
    const synthetic = r.source === "synthetic", active = data.running && data.active_id === data.selected_id;
    $("status").textContent = statuses[r.status] || r.status;
    if (!active && r.status !== "CLOSED") $("status").textContent = "Сессия сохранена · остановлена";
    $("status-dot").className = `status-dot ${active && r.status === "RUNNING" ? "active" : r.status !== "CLOSED" ? "warning" : ""}`;
    $("source").textContent = sourceNames[r.source] || r.source;
    $("session-note").textContent = r.status === "UNRESOLVED" ? r.broker?.error || "Остаток или исполнение не подтверждены. Восстанови сессию для сверки." : synthetic ? "Ускоренное модельное время. Результат демо не оценивает доходность." : r.source === "alpaca-paper" ? "Виртуальные ордера в Alpaca. Сервер должен оставаться включён; комиссии могут начисляться позже." : "Виртуальные сделки на текущих наблюдениях. Сервер должен оставаться включён.";
    if(data.health?.recovery_required) $("session-note").textContent=`Последний сохранённый шаг: ${new Date(state.last_tick_at || state.started_at).toLocaleString("ru-RU")}. Исполнитель в этой панели не запущен. Восстановление выполняется отдельно; исходный дедлайн сохраняется.`;
    $("equity").textContent = money(r.equity_at_last_quote);
    $("cash").textContent = `Свободно: ${money(r.cash)}`;
    $("pnl").textContent = `${Number(r.net_pnl_at_last_quote) > 0 ? "+" : ""}${money(r.net_pnl_at_last_quote)}`;
    colorValue("pnl", r.net_pnl_at_last_quote);
    $("pnl-percent").textContent = `${number(Number(r.net_pnl_at_last_quote) / Number(r.capital) * 100)} % · ${r.entries} входов / ${r.exits} выходов`;
    $("fee-label").textContent = r.fees_are_estimated ? "Оценка комиссий" : "Комиссии";
    $("fees").textContent = money(r.fees); $("drawdown").textContent = `Макс. просадка: ${number(Number(r.max_drawdown) * 100)} %`;
    const quote = state.last_quote;
    $("price").textContent = quote ? money((Number(quote.bid) + Number(quote.ask)) / 2) : "—";
    $("quote-time").textContent = quote ? `Наблюдение: ${time(quote.observed_at)}${synthetic ? " · модель" : ""}` : "Ожидаем котировку";
    $("quantity").textContent = `${number(r.quantity,8)} BTC`;
    $("entry").textContent = Number(r.quantity) > 0 ? money(state.entry_price) : "—";
    $("stop-price").textContent = Number(r.quantity) > 0 ? money(state.stop_price) : "—";
    if(r.exit_policy === "trailing-stop-v1" && Number(r.quantity)>0) $("stop-price").textContent+=` · ${number(r.trailing_percent)} % от максимума ${money(r.position_peak)}`;
    $("unrealized").textContent = money(r.unrealized_pnl_at_last_quote); colorValue("unrealized",r.unrealized_pnl_at_last_quote);
    $("position-badge").textContent = Number(r.quantity) > 0 ? "Виртуальная позиция" : "Нет позиции";
    $("chart-caption").textContent = `${data.points.length} наблюдений · ${synthetic ? "синтетические данные" : r.source === "alpaca-paper" ? "котировки Alpaca" : "опрос публичного источника"}`;
    $("event-count").textContent = data.event_count;
    const pending = r.broker?.pending;
    $("broker-order").hidden = !r.broker;
    $("broker-order").textContent = pending ? "Заявка: " + (pending.request.side === "buy" ? "покупка" : "продажа") + " · " + pending.request.qty + " BTC · " + pending.phase + " · исполнено " + pending.filled_qty + " BTC" : r.broker ? "Заявок в обработке нет. Сверка: " + time(r.broker.last_reconcile_at) : "";
  }
  if (!r) resetOverview();
  if (data.worker_error) alertMessage(data.worker_error);
  renderClock(); renderButtons(); renderEvents(); renderVerification(data);
  priceChart.update(data); drawChart("equity-chart", data.points, "equity", []);
  $("price-empty").hidden = data.points.length > 0; $("equity-empty").hidden = data.points.length > 0;
}
function renderClock() {
  if (!current?.state) return;
  const s = current.state, synthetic = current.report.source === "synthetic";
  const now = synthetic || !current.running || current.selected_id !== current.active_id ? new Date(s.last_tick_at || s.started_at).getTime() : Date.now();
  const end = new Date(s.deadline_at).getTime(), start = new Date(s.started_at).getTime();
  const left = Math.max(0, Math.ceil((end-now)/1000));
  $("countdown").textContent = s.status === "CLOSED" ? "Завершено" : `${String(Math.floor(left/3600)).padStart(2,"0")}:${String(Math.floor(left%3600/60)).padStart(2,"0")}:${String(left%60).padStart(2,"0")}`;
  $("clock-label").textContent = synthetic ? "Ускоренное модельное время" : `Дедлайн: ${time(s.deadline_at)}`;
  $("progress").value = s.status === "CLOSED" ? 100 : Math.max(0,Math.min(100,(now-start)/(end-start)*100));
}
function renderVerification(data) {
  renderHealth(data);
  $("decision-flow").replaceChildren();
  const execution={NONE:"Исполнения нет",PREPARED:"Подготовлено",NEW:"Принято брокером",PARTIALLY_FILLED:"Частичное исполнение",FILLED:"Исполнено",UNKNOWN:"Результат неизвестен",REJECTED:"Отклонено",ABORTED:"Отменено до отправки",CANCELED:"Отменено",CANCEL_REQUESTED:"Отмена запрошена",EXPIRED:"Истекло"};
  for(const item of (data.decision_flow || []).slice(-8).reverse()) {
    const row=document.createElement("tr");
    const action={WAIT:"Ждать",REQUEST_ENTRY:"Вход",REQUEST_EXIT:"Выход"}[item.action] || item.action;
    for(const value of [time(item.time),`${action} · ${item.origin === "QWEN" ? "Qwen" : item.origin === "CONTROL" ? "Защитный контроллер" : "Правило"} · ${reasons[item.reason] || item.reason}`,item.risk ? `${item.risk.approved ? "Разрешено" : "Отказ"} · ${reasons[item.risk.reason] || item.risk.reason}` : "Проверка ордера не нужна",`${execution[item.execution] || item.execution}${Number(item.filled_quantity)>0 ? " · " + number(item.filled_quantity,8)+" BTC" : ""}`]) {const td=document.createElement("td");td.textContent=value;row.append(td);}
    $("decision-flow").append(row);
  }
  const report=data.comparison;
  $("comparison-results").replaceChildren();
  if(report) for(const [key,label] of [["usd","Хранение USD"],["demo_momentum","Тестовое правило"],["recorded_qwen","Записанная Qwen"]]) {
    const arm=report.results[key],row=document.createElement("tr");
    for(const value of [label,money(arm.net_pnl),money(arm.fees),number(Number(arm.max_drawdown)*100)+" %"]) {const td=document.createElement("td");td.textContent=value;row.append(td);} $("comparison-results").append(row);
  }
  $("comparison-note").textContent=data.maintenance?.running ? "Выполняется проверка…" : data.maintenance?.error || (report ? `Одинаковые данные и лимиты, моделируемые расходы. Покрытие записанных ответов Qwen: ${number(Number(report.qwen_replay.coverage)*100)} %. Преимущество не подтверждено; независимый прогон: команда experiment.` : "После закрытия можно воспроизвести сессию в локальном симуляторе.");
  const accounting=data.report?.accounting;
  $("accounting-note").textContent=data.config?.source === "alpaca-paper" ? accounting?.status ? `${accounting.status === "RECONCILED_AS_OF" ? "Начисления сверены" : "Сверка не подтверждена"} на ${time(accounting.checked_at)} · комиссии по начислениям ${money(accounting.fees_usd)}${accounting.unallocated_activity_ids.length ? " · неоднозначных начислений: " + accounting.unallocated_activity_ids.length : ""}. Итог может уточняться при новых начислениях.` : "Комиссии оценочные. Начисления Alpaca могут появиться после торгового дня." : "Локальные комиссии моделируются по параметрам сессии.";
  $("reconcile-fees").hidden=data.config?.source !== "alpaca-paper";
  if(accounting?.reason_codes?.length) $("accounting-note").textContent+=` Причины: ${accounting.reason_codes.join(", ")}.`;
  if(accounting?.actual_cash != null) $("accounting-note").textContent+=` По чтению счёта Alpaca: ${money(accounting.actual_cash)} · ${number(accounting.actual_quantity,9)} BTC.`;
  if(data.config?.source === "alpaca-paper" && data.report?.status === "UNRESOLVED") $("accounting-note").textContent+=" Кнопка сверки читает данные счёта и начисления, не возобновляя торговлю.";
}
function toast(message) {
  clearTimeout(toastTimer); $("toast").textContent = message; $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4000);
}
function renderSessions(data) {
  const query = $("session-search").value.trim().toLocaleLowerCase("ru-RU"), filter = $("session-filter").value;
  const label = s => `${sourceNames[s.source] || s.source} · ${new Date(s.started_at).toLocaleString("ru-RU", {year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"})} · ${s.status === "CLOSED" ? "завершена" : "открыта"}`;
  const matches = data.sessions.filter(s => (filter === "all" || (filter === "closed" ? s.status === "CLOSED" : s.status !== "CLOSED")) && (!query || `${label(s)} ${s.session_id} ${s.status} ${s.source}`.toLocaleLowerCase("ru-RU").includes(query)));
  const options = matches.map(s => ({value:s.session_id,label:label(s)}));
  const chosen = data.sessions.find(s => s.session_id === data.selected_id);
  const outside = chosen && !matches.some(s => s.session_id === chosen.session_id);
  if (outside) options.unshift({value:chosen.session_id,label:`Выбрана · ${label(chosen)}`});
  const signature = JSON.stringify(options);
  if ($("sessions").dataset.signature !== signature) {
    $("sessions").replaceChildren();
    for (const option of options) { const node = document.createElement("option"); node.value = option.value; node.textContent = option.label; $("sessions").append(node); }
    if (!options.length) { const node = document.createElement("option"); node.value = ""; node.textContent = data.sessions.length ? "Ничего не найдено" : "Пока нет сессий"; $("sessions").append(node); }
    $("sessions").dataset.signature = signature;
  }
  $("sessions").disabled = !options.length || busy;
  if (data.selected_id) $("sessions").value = data.selected_id;
  $("session-id").textContent = data.selected_id || "—";
  $("session-count").textContent = `${matches.length} / ${data.sessions.length}`;
  $("session-search-note").textContent = outside ? `Выбранная сессия вне фильтра. ${matches.length ? "Совпадения перечислены ниже неё." : "Совпадений нет."}` : "";
}
function resetOverview() {
  for (const id of ["equity","pnl","fees","quantity","entry","stop-price","unrealized","price","countdown"]) { $(id).textContent = "—"; $(id).classList.remove("positive","negative"); }
  $("status").textContent = "Готов к запуску"; $("status-dot").className = "status-dot"; $("source").textContent = "Нет активной сессии";
  $("session-note").textContent = "Запусти новую сессию или выбери сохранённую.";
  $("cash").textContent = "Свободные средства: —"; $("pnl-percent").textContent = "После расходов исполнения";
  $("drawdown").textContent = "Макс. просадка: —"; $("fee-label").textContent = "Комиссии";
  $("clock-label").textContent = "Время сессии"; $("progress").value = 0; $("event-count").textContent = 0;
  $("quote-time").textContent = "Котировок пока нет"; $("chart-caption").textContent = "—";
  $("position-badge").textContent = "Нет позиции"; $("broker-order").hidden = true;
}
function eventSearchText(event) {
  const labels = {PAPER_FILL:"сделка",BROKER_FILL:"сделка",AI_REQUEST:"запрос Qwen",AI_DECISION:"ответ Qwen",SNAPSHOT:"котировка",SESSION_STARTED:"старт сессии",SESSION_STATUS:"состояние",DECISION:"решение",RISK_VERDICT:"проверка риска",TRAILING_STOP_MOVED:"скользящий стоп поднят",ORDER_PREPARED:"заявка подготовлена",ORDER_UPDATE:"состояние заявки",ORDER_UNKNOWN:"ответ не подтверждён",ORDER_REJECTED:"заявка отклонена",ORDER_CANCEL_REQUESTED:"запрос отмены",ORDER_ABORTED:"вход отменён",BROKER_LEDGER:"баланс сверен",DATA_UNAVAILABLE:"нет данных",DATA_REJECTED:"данные отклонены",DECISION_UNAVAILABLE:"нет решения"};
  const p = event.payload;
  return `${event.kind} ${labels[event.kind] || ""} ${p.side === "BUY" ? "покупка" : p.side === "SELL" ? "продажа" : ""} ${reasons[p.reason] || ""} ${statuses[p.to] || ""} ${JSON.stringify(p)}`;
}
function resetLog() { logOffset = 0; logAnchor = null; $("events").closest(".table-scroll").scrollTop = 0; renderEvents(); }
function renderEvents() {
  if (logSession !== current?.selected_id) { logSession = current?.selected_id; logOffset = 0; logAnchor = null; }
  const important = new Set(["PAPER_FILL","SESSION_STARTED","SESSION_STATUS","DATA_UNAVAILABLE","DATA_REJECTED","DECISION_UNAVAILABLE","BROKER_FILL","ORDER_PREPARED","ORDER_UPDATE","ORDER_UNKNOWN","ORDER_REJECTED","BROKER_UNRESOLVED","ORDER_CANCEL_REQUESTED","ORDER_ABORTED","ORDER_ABSENT_CONFIRMED","WORKER_ERROR","WATCHDOG_TRIPPED"]);
  const mode = $("log-filter").value;
  const source = mode === "blockers" ? current?.blocking_events || [] : mode === "all" ? current?.events || [] : current?.important_events || (current?.events || []).filter(e=>important.has(e.kind));
  const query = $("log-search").value.trim().toLocaleLowerCase("ru-RU");
  const matches = [...source].reverse().filter(e => !query || [time(e.occurred_at), eventSearchText(e)].join(" ").toLocaleLowerCase("ru-RU").includes(query));
  if (logOffset > 0 && logAnchor != null) { const at = matches.findIndex(e => e.seq === logAnchor); if (at >= 0) logOffset = at; }
  logOffset = Math.max(0, Math.min(logOffset, Math.max(0, matches.length - 1)));
  const events = matches.slice(logOffset, logOffset + 25);
  logAnchor = logOffset > 0 ? events[0]?.seq : null;
  $("log-prev").disabled = logOffset === 0;
  $("log-next").disabled = logOffset + 25 >= matches.length;
  $("log-page").textContent = matches.length ? `${logOffset + 1}–${Math.min(logOffset + 25, matches.length)} из ${matches.length}` : "Найдено: 0";
  $("log-scope").textContent = mode === "blockers" ? "Последние 200 записей: направление, источник и причина. Повторы подряд объединены. Удержание и ожидание отделены от отказов. Полный список — в отчёте." : mode === "all" ? "Поиск среди последних 500 событий; по 25 на странице. Полный журнал — в отчёте." : "Поиск среди последних 50 событий сделок и состояния; по 25 на странице. Полный журнал — в отчёте.";
  if (mode === "blockers" && current && !(current.capabilities || []).includes("trade-blocking-log-v1")) $("log-scope").textContent="Этот сервер ещё не поддерживает список блокировок. Откройте панель через launch-dashboard.cmd.";
  const scroll = $("events").closest(".table-scroll"), scrollTop = scroll.scrollTop;
  $("events").replaceChildren();
  if (!events.length) { const row = document.createElement("tr"), cell = document.createElement("td"); cell.colSpan=5; cell.className="empty-row"; cell.textContent=query ? "По этому запросу событий нет." : "Пока нет событий. Наблюдения записываются во время сессии."; row.append(cell); $("events").append(row); return; }
  for (const event of events) {
    const p = event.payload, row = document.createElement("tr");
    const cells = Array.from({length:5},()=>document.createElement("td"));
    cells[0].textContent = time(event.occurred_at);
    const components = event.kind === "TRADE_BLOCKER" ? [p.component] : (current?.blocking_events || [])
      .filter(item => event.seq != null && item.payload.source_seq === event.seq && item.payload.source_kind === event.kind)
      .map(item => item.payload.component);
    cells[2].textContent = [...new Set(components.filter(Boolean))].join(" · ") || (event.kind === "TRADE_BLOCKER" ? "Не указан" : "—");
    cells[2].className = "blocker-source";
    let label = {ORDER_ABSENT_CONFIRMED:"Отсутствие заявки сверено", ORDER_ABSENCE_CHECK:"Проверка отсутствия заявки", BROKER_READY:"Alpaca готова", ORDER_PREPARED:"Заявка подготовлена", ORDER_SUBMITTING:"Отправка заявки", ORDER_UPDATE:"Состояние заявки", ORDER_UNKNOWN:"Ответ не подтверждён", ORDER_REJECTED:"Заявка отклонена", BROKER_UNRESOLVED:"Нужна сверка", ORDER_CANCEL_REQUESTED:"Запрос отмены", ORDER_ABORTED:"Вход отменён", BROKER_LEDGER:"Баланс сверен", SESSION_STARTED:"Старт сессии",SESSION_STATUS:"Состояние",SNAPSHOT:"Котировка",DECISION:"Решение",RISK_VERDICT:"Проверка риска",DATA_UNAVAILABLE:"Нет данных",DATA_REJECTED:"Данные отклонены",DECISION_UNAVAILABLE:"Нет решения"}[event.kind] || event.kind;
    if (["PAPER_FILL","BROKER_FILL"].includes(event.kind)) label = p.side === "BUY" ? "Покупка" : "Продажа";
    if (event.kind === "AI_REQUEST") label = "Запрос Qwen";
    if (event.kind === "BROKER_RECONCILIATION_MISMATCH") label="Расхождение баланса или позиции";
    if (event.kind === "LATE_FEES_RECONCILED") label="Начисления объяснили расхождение";
    if (event.kind === "AI_DECISION") label = p.accepted ? "Ответ Qwen" : "Ответ Qwen отклонён";
    if (event.kind === "TRAILING_STOP_MOVED") label="Скользящий стоп поднят";
    if (event.kind === "TRADE_BLOCKER") label=p.scope;
    const badge = document.createElement("span"); badge.className=`event-tag ${["PAPER_FILL","BROKER_FILL"].includes(event.kind) ? p.side === "BUY" ? "buy" : "sell" : ""}`; badge.textContent=label; cells[1].append(badge);
    if (event.kind === "TRADE_BLOCKER") cells[3].textContent=`${reasons[p.code] || p.detail}${(reasons[p.code] || p.detail) === p.code ? "" : " · " + p.code}${Object.keys(p.facts || {}).length ? " · " + JSON.stringify(p.facts) : ""} · исходное событие ${p.source_kind} #${p.source_seq ?? "—"}${p.count > 1 ? ` · повторов: ${p.count}, с ${time(p.first_at)}` : ""}`;
    else if (event.kind === "SNAPSHOT") cells[3].textContent=`bid ${money(p.bid)} / ask ${money(p.ask)}`;
    else if (event.kind === "SESSION_STARTED") cells[3].textContent=`Виртуальный бюджет ${money(p.capital)}`;
    else if (event.kind === "SESSION_STATUS") cells[3].textContent=`${statuses[p.to] || p.to}${p.reason ? ` · ${reasons[p.reason] || p.reason}` : ""}`;
    else if (event.kind === "ORDER_UPDATE") cells[3].textContent = p.side + " · " + p.status + " · исполнено " + p.filled_qty + " BTC";
    else if (event.kind === "AI_REQUEST") cells[3].textContent = "Подготовленный снимок " + p.snapshot.snapshot_id.slice(0,12);
    else if (event.kind === "AI_DECISION") cells[3].textContent = ({WAIT:"Ждать",REQUEST_ENTRY:"Войти",REQUEST_EXIT:"Выйти"}[p.action] || p.action) + " · " + p.reason;
    else if (event.kind === "TRAILING_STOP_MOVED") cells[3].textContent=`${money(p.previous_stop)} → ${money(p.stop_price)} · максимум ${money(p.position_peak)}`;
    else cells[3].textContent=reasons[p.reason] || p.reason || "Проверка входных данных";
    if (["PAPER_FILL","BROKER_FILL"].includes(event.kind)) cells[3].textContent=`${number(p.quantity,6)} BTC × ${money(p.price)} · ${reasons[p.reason] || "Допустимый тестовый вход"}`;
    cells[4].className="numeric"; cells[4].textContent=p.realized_pnl !== undefined ? money(p.realized_pnl) : event.kind === "RISK_VERDICT" ? p.approved ? "Разрешено" : "Отказ" : "—";
    if (event.kind === "TRADE_BLOCKER") cells[4].textContent=p.status;
    if (p.realized_pnl !== undefined) cells[4].classList.add(Number(p.realized_pnl)>=0 ? "positive" : "negative");
    row.append(...cells); $("events").append(row);
  }
  scroll.scrollTop = scrollTop;
}
const svgNS="http://www.w3.org/2000/svg";
function svgNode(tag,attrs,text) { const node=document.createElementNS(svgNS,tag); for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value); if(text!==undefined)node.textContent=text; return node; }
function drawChart(id, points, key, fills) {
  const svg=$(id); svg.replaceChildren(); if(!points.length)return;
  const width=Math.max(280,svg.getBoundingClientRect().width), height=key==="price"?260:180, left=74,right=16,top=20,bottom=30;
  svg.setAttribute("viewBox",`0 0 ${width} ${height}`);
  const values=points.map(p=>Number(p[key])); let low=Math.min(...values),high=Math.max(...values);
  const pad=Math.max((high-low)*.14,high*.001,.01); low-=pad; high+=pad;
  const first=new Date(points[0].time).getTime(), last=new Date(points[points.length-1].time).getTime();
  const x=t=>left+(new Date(t).getTime()-first)/Math.max(1,last-first)*(width-left-right), y=v=>top+(high-v)/(high-low)*(height-top-bottom);
  for(let i=0;i<4;i++) { const v=low+(high-low)*i/3, yy=y(v); svg.append(svgNode("line",{x1:left,x2:width-right,y1:yy,y2:yy,class:"grid"})); svg.append(svgNode("text",{x:left-8,y:yy+3,"text-anchor":"end"},number(v))); }
  const coords=points.map(p=>`${x(p.time)},${y(Number(p[key]))}`), path=`M${coords.join(" L")}`;
  svg.append(svgNode("path",{d:`${path} L${x(points[points.length-1].time)},${height-bottom} L${left},${height-bottom} Z`,class:key==="price"?"area":"area equity-area"}));
  svg.append(svgNode("path",{d:path,class:key==="price"?"line":"line equity-line"}));
  if(points.length===1)svg.append(svgNode("circle",{cx:left,cy:y(values[0]),r:3,fill:"#7ce3c1"}));
  svg.append(svgNode("text",{x:left,y:height-8},time(points[0].time)));
  svg.append(svgNode("text",{x:width-right,y:height-8,"text-anchor":"end"},time(points[points.length-1].time)));
  for(const fill of fills) { const nearest=points.reduce((best,p)=>Math.abs(new Date(p.time)-new Date(fill.time))<Math.abs(new Date(best.time)-new Date(fill.time))?p:best,points[0]); const circle=svgNode("circle",{cx:x(fill.time),cy:y(Number(nearest[key])),r:4.5,fill:fill.side==="BUY"?"#7ce3c1":"#edc080",stroke:"#141d26","stroke-width":2});circle.append(svgNode("title",{},`${fill.side==="BUY"?"Покупка":"Продажа"} ${time(fill.time)} · ${money(fill.price)}`));svg.append(circle); }
  const overlay=svgNode("rect",{x:left,y:top,width:width-left-right,height:height-top-bottom,fill:"transparent"});
  const focus=svgNode("circle",{cx:left,cy:y(values[0]),r:4,fill:"#e8eff6",visibility:"hidden"});
  const tooltip=svgNode("text",{x:left+6,y:12}); svg.append(overlay,focus,tooltip);
  overlay.addEventListener("pointermove",event=>{const pos=(event.clientX-svg.getBoundingClientRect().left)/svg.getBoundingClientRect().width*width; const p=points.reduce((best,p)=>Math.abs(x(p.time)-pos)<Math.abs(x(best.time)-pos)?p:best,points[0]);focus.setAttribute("cx",x(p.time));focus.setAttribute("cy",y(Number(p[key])));focus.setAttribute("visibility","visible");tooltip.textContent=`${time(p.time)} · ${money(p[key])}`;});
  overlay.addEventListener("pointerleave",()=>{focus.setAttribute("visibility","hidden");tooltip.textContent="";});
}
function modeHelp() {
  const mode = $("mode").value;
  if (mode === "demo" || (mode !== "alpaca" && $("strategy").value === "smoke")) $("strategy").value = "momentum";
  $("mode-help").textContent = $("strategy").value === "qwen" ? "Qwen выбирает вход по подготовленным данным; открытую позицию сопровождает скользящий стоп. Цель необязательна, риск и дедлайн фиксированы. Только виртуальные средства." : mode === "demo" ? "Ускоренная демонстрация. Не бэктест и не оценка доходности." : mode === "alpaca" ? $("strategy").value === "smoke" ? "Создаст один виртуальный вход и выход через Alpaca, затем завершит сессию. Это проверка механики, без ИИ." : "Исполнение в Alpaca Paper. Нужен счёт без позиций и открытых заявок. Решения принимает тестовое правило." : "Текущие котировки BTC/USD и собственный симулятор исполнения.";
  renderButtons();
}
$("mode").addEventListener("change",()=>{ $("duration").value = $("mode").value === "demo" ? 60 : 240; modeHelp(); });
$("strategy").addEventListener("change",modeHelp);
$("llm-model").addEventListener("change",()=>{modelChoiceInitialized=true;if(current)renderModel(current);renderButtons();});
$("model-connect").addEventListener("click",async()=>{try{await api("/api/model/connect",{model:$("llm-model").value});await refresh();}catch(error){alertMessage(error.message);}});
$("settings").addEventListener("submit",event=>{event.preventDefault();command("/api/start",Object.fromEntries(new FormData(event.target)));});
$("sessions").addEventListener("change",()=>{selected=$("sessions").value;alertMessage("");refresh();});
$("session-search").addEventListener("input",()=>{if(current)renderSessions(current);});
$("session-filter").addEventListener("change",()=>{if(current)renderSessions(current);});
$("active-session").addEventListener("click",()=>{selected=current.active_id;alertMessage("");refresh();});
$("copy-session").addEventListener("click",async()=>{try{await navigator.clipboard.writeText(current.selected_id);toast("ID сессии скопирован");}catch{alertMessage("Не удалось скопировать ID. Его можно выделить в истории сессий.");}});
$("delete-session").addEventListener("click",()=>{
  if($("delete-session").disabled)return;
  deleteTarget=current.selected_id;
  $("delete-description").textContent=`${sourceNames[current.config.source] || current.config.source} · ${new Date(current.state.started_at).toLocaleString("ru-RU")} · ID ${deleteTarget}`;
  $("delete-dialog").returnValue="cancel";
  $("delete-dialog").showModal();
});
$("delete-dialog").addEventListener("close",async()=>{
  const target=deleteTarget;deleteTarget=null;
  if($("delete-dialog").returnValue!=="delete" || !target || busy)return;
  busy=true;alertMessage("");renderButtons();
  try{await api("/api/sessions/delete",{session_id:target});selected=null;await refresh();toast("История сессии удалена");}
  catch(error){alertMessage(error.message);}
  finally{busy=false;renderButtons();}
});
$("pause").addEventListener("click",()=>command("/api/control",{session_id:current.selected_id,action:current.report.status==="PAUSED_ENTRIES"?"continue":"pause"}));
$("stop").addEventListener("click",()=>command("/api/control",{session_id:current.selected_id,action:"stop"}));
$("restore").addEventListener("click",()=>command("/api/resume",{session_id:current.selected_id}));
$("compare").addEventListener("click",()=>command("/api/compare",{session_id:current.selected_id}));
$("reconcile-fees").addEventListener("click",()=>command("/api/accounting/reconcile",{session_id:current.selected_id}));
$("log-filter").addEventListener("change",resetLog);
$("log-search").addEventListener("input",resetLog);
$("log-prev").addEventListener("click",()=>{logOffset=Math.max(0,logOffset-25);logAnchor=null;$("events").closest(".table-scroll").scrollTop=0;renderEvents();});
$("log-next").addEventListener("click",()=>{logOffset+=25;logAnchor=null;$("events").closest(".table-scroll").scrollTop=0;renderEvents();});
$("export").addEventListener("click",async()=>{try{const data=await api(`/api/export?session=${encodeURIComponent(current.selected_id)}`);const blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=`traid-${current.selected_id}.json`;a.click();URL.revokeObjectURL(url);}catch(error){alertMessage(error.message);}});
function renderHealth(data) {
  const health = data.health;
  renderDecisionSummary(data);
  const labels = {READY:"Готово",READING:"Чтение",TIMED_OUT:"Время ожидания истекло",ERROR:"Ошибка",UNCONFIRMED:"Не подтверждено",UNAVAILABLE:"Нет данных",STALE:"Устарело",ARCHIVE:"Архив",SIMULATED:"Демо",NOT_USED:"Не используется",PENDING:"Ожидаем ответ",REJECTED:"Ответ не принят",DISCARDED:"Ответ отменён",WARMUP:"Собираем данные",GAPS:"Пропуски",WAITING:"Ожидание",STOPPED:"Остановлено"};
  const stages = {starting:"начало чтения",order:"поиск заявки",account:"баланс счёта",positions:"позиции",open_orders:"открытые заявки",order_history:"история заявок"};
  const archive = health?.scope === "ARCHIVE";
  Object.assign(labels,{RECOVERY_REQUIRED:"Требуется восстановление",INTERRUPTED:"Запрос прерван"});
  $("health-status").textContent = health?.recovery_required ? "Требуется восстановление" : archive ? "Архив сессии" : health?.scope === "LIVE" ? "Наблюдение" : "Нет активной сессии";
  $("health-details").textContent = archive ? `${health.recovery_required ? "Сессия не завершена. Автоматического продолжения нет. " : ""}Показатели на последний шаг сессии: ${time(health.as_of)}. Возраст данных рассчитан на тот момент.` : `Состояние на ${time(health?.as_of)}. ${data.watchdog?.tripped ? "Watchdog: " + (reasons[data.watchdog.reason] || data.watchdog.reason) : "Обновляется автоматически."}`;
  $("health-components").replaceChildren();
  for (const item of health?.components || []) {
    const card = document.createElement("article"); card.className = "health-card";
    const label = document.createElement("span"); label.textContent = item.label;
    const status = document.createElement("strong"); status.textContent = labels[item.status] || item.status;
    if (["ERROR","TIMED_OUT","STALE","REJECTED","RECOVERY_REQUIRED","INTERRUPTED"].includes(item.status)) status.className = "negative";
    const details = [item.detail];
    if (item.age_seconds != null) details.push(`Возраст: ${number(item.age_seconds,1)} с`);
    if (item.last_success_at) details.push(`Успешная сверка: ${time(item.last_success_at)}`);
    if (item.last_progress_at) details.push(`Последний шаг: ${time(item.last_progress_at)}`);
    if (item.count != null) details.push(`Свечей: ${item.count}`);
    if (item.waiting_seconds != null) details.push(`Ожидание: ${number(item.waiting_seconds,1)} с`);
    if (item.review_at) details.push(`Пересмотр по следующей закрытой свече или в ${time(item.review_at)}`);
    if (item.wait_streak) details.push(`Последовательных отсрочек: ${item.wait_streak}`);
    if (item.latency_ms != null) details.push(`Последний ответ: ${number(item.latency_ms/1000,2)} с`);
    if (item.error_code) details.push(`Причина: ${item.error_code}`);
    if (item.elapsed_ms != null) details.push(`Полное задание: ${number(item.elapsed_ms/1000,2)} с`);
    if (item.result_wait_ms != null) details.push(`После готовности: ${number(item.result_wait_ms/1000,2)} с`);
    if (item.pending_phase) details.push(`Заявка: ${item.pending_phase}`);
    if (item.reconciliation?.status === "MISMATCH") {
      const r=item.reconciliation;
      details.push(`Чтение ${time(r.checked_at)}: ${money(r.actual_cash)}, ${number(r.actual_quantity,9)} BTC`);
      details.push(`Ожидалось USD ${money(r.expected_cash_low)}–${money(r.expected_cash_high)}; BTC ${number(r.expected_quantity_low,9)}–${number(r.expected_quantity_high,9)}`);
    }
    if (item.io?.stage) details.push(`Запрос: ${stages[item.io.stage] || item.io.stage} · ${number(item.io.elapsed_seconds,1)} / ${item.io.budget_seconds} с`);
    if (item.io?.error) details.push(item.io.error);
    if (item.io?.duration_seconds != null) details.push(`Загрузка: ${number(item.io.duration_seconds,2)} с`);
    const text = document.createElement("p"); text.textContent = details.filter(Boolean).join(" · ");
    card.append(label,status,text); $("health-components").append(card);
  }
  $("health-blockers").textContent = health?.blockers?.length ? "Новые входы: " + health.blockers.map(b=>b.text).join("; ") + "." : health?.scope === "LIVE" ? "Технических препятствий по этим показателям не обнаружено. Каждый ордер отдельно проверяет Risk Engine." : "";
  const decision = health?.latest_decision;
  $("health-decision").textContent = decision ? `Последнее торговое решение (${time(decision.time)}): ${{WAIT:"ждать",REQUEST_ENTRY:"предложить вход",REQUEST_EXIT:"предложить выход"}[decision.action] || decision.action} · ${reasons[decision.reason] || decision.reason}` : "Торговых решений пока нет.";
}
function renderDecisionSummary(data) {
  const summary=data.decision_summary;
  $("decision-summary").hidden=!summary?.requests;
  if (!summary?.requests) return;
  $("decision-summary-counts").textContent=`Запросов ИИ: ${summary.requests} · вход был доступен в ${summary.requests_with_entry_option} снимках · принятых предложений входа: ${summary.accepted_entries} · принятых ответов «ждать»: ${summary.accepted_waits} · ответов не принято: ${summary.rejected} · без сохранённого ответа: ${summary.unanswered}.`;
  $("decision-summary-blocks").replaceChildren();
  const factLabels={c15_range_pct:"диапазон 15 мин, %",round_trip_cost_pct:"расходы, %",c5_return_pct:"изменение 5 мин, %",c15_return_pct:"изменение 15 мин, %",c5_volume_btc:"объём 5 мин, BTC",seconds_to_flatten:"до закрытия, с"};
  for(const item of [...summary.blocking_conditions,...(summary.warnings || []).map(w=>({...w,warning:true}))]) {
    const text=document.createElement("p"); text.className="field-help";
    const facts=Object.entries(item.latest_facts).map(([key,value])=>`${factLabels[key] || key}: ${value ?? "нет данных"}`).join("; ");
    text.textContent=`${item.warning ? "Предупреждение, не запрет: " : ""}${item.label}: ${item.count} снимков. Последние значения: ${facts}.`;
    $("decision-summary-blocks").append(text);
  }
  const refusalLabels={BROKER_NOT_READY:"Брокер не готов: требуется сверка или завершение заявки",BUDGET_OR_MINIMUM:"Бюджет или минимальный размер ордера"};
  for (const [group,counts] of [["Отказ исполнения",summary.risk_reason_counts],["Ответ ИИ отклонён",summary.rejected_reason_counts]]) {
    for (const [reason,count] of Object.entries(counts || {})) {
      const text=document.createElement("p"); text.className="field-help";
      text.textContent=`${group}: ${refusalLabels[reason] || reason} · ${count}.`;
      $("decision-summary-blocks").append(text);
    }
  }
  $("decision-summary-note").textContent=summary.blocked && !summary.requests_with_entry_option ? "На этих снимках вход исключали условия кода. Модель не могла выбрать покупку. Диапазон описывает прошлое; сравнение с расходами — исследовательский фильтр." : `Отказов риска или готовности исполнителя: ${summary.risk_denials} · событий исполнения: ${summary.fills}. Предложение ИИ не означает отправку ордера. Сводка построена по сохранённым снимкам и ответам.`;
}
modeHelp(); refresh(); setInterval(refresh,1000); setInterval(renderClock,1000);
