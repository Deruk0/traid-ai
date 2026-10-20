"use strict";
let alpacaBusy = false, alpacaFetching = false;
let alpacaCommandError = null;
const alpacaOrderStatuses = {new:"Принята",accepted:"Принята",pending_new:"Принимается",
  partially_filled:"Частично исполнена",filled:"Исполнена",canceled:"Отменена",
  pending_cancel:"Отменяется",expired:"Истекла",rejected:"Отклонена",replaced:"Заменена",
  pending_replace:"Заменяется",done_for_day:"Завершена на сегодня",held:"Ожидает"};
function alpacaButtons() {
  $("alpaca-connect-button").disabled = alpacaBusy || !token;
  $("alpaca-env").disabled = alpacaBusy || !token;
  $("alpaca-disconnect").disabled = alpacaBusy || !token || (current?.running && current.execution_source === "alpaca-paper");
  $("alpaca-key").disabled = alpacaBusy; $("alpaca-secret").disabled = alpacaBusy;
}
function alpacaTable(id, rows) {
  const body = $(id); body.replaceChildren();
  if (!rows.length) {
    const row = document.createElement("tr"), cell = document.createElement("td");
    cell.colSpan = 5; cell.className = "empty-row"; cell.textContent = "Записей пока нет";
    row.append(cell); body.append(row); return;
  }
  for (const values of rows) {
    const row = document.createElement("tr");
    for (const value of values) { const cell = document.createElement("td"); cell.textContent = value ?? "—"; row.append(cell); }
    body.append(row);
  }
}
function renderAlpaca(data) {
  $("alpaca-status").textContent = !data.connected ? "Не подключена" : Object.keys(data.errors).length ? "Ошибка обновления" : data.refreshing ? "Получаем данные…" : "Paper подключена";
  $("alpaca-connect").hidden = data.connected;
  $("alpaca-env").hidden = !data.environment_keys_available;
  $("alpaca-help").hidden = data.connected;
  $("alpaca-account").hidden = !data.connected;
  const labels = {account:"Баланс",positions:"Позиции",orders:"Заявки",quote:"Котировка"};
  const failures = Object.entries(data.errors).map(([key, message]) => labels[key] + ": " + message + ". Показаны последние полученные данные.");
  if (alpacaCommandError) failures.unshift(alpacaCommandError);
  $("alpaca-error").textContent = failures.join(" "); $("alpaca-error").hidden = !failures.length;
  if (data.connected) {
    const a = data.account;
    $("alpaca-equity").textContent = money(a.equity); $("alpaca-cash").textContent = money(a.cash);
    $("alpaca-power").textContent = money(a.non_marginable_buying_power ?? a.cash);
    const q = data.quote;
    $("alpaca-quote").textContent = q ? number(q.bid) + " / " + number(q.ask) : "—";
    $("alpaca-quote").classList.toggle("negative", Boolean(q?.stale));
    $("alpaca-updated").textContent = "Счёт …" + a.account_suffix + " · " + (a.status === "ACTIVE" ? "Активен" : a.status) +
      (a.trading_blocked || a.account_blocked ? " · операции ограничены Alpaca" : "") +
      " · баланс: " + time(data.resource_times.account) + " · позиции: " + time(data.resource_times.positions) +
      " · заявки: " + time(data.resource_times.orders) +
      (q ? " · котировка: " + time(q.observed_at) + (q.stale ? " (устарела)" : "") : "");
    alpacaTable("alpaca-positions", data.positions.map(p => [p.symbol, p.qty, p.avg_entry_price, p.market_value, p.unrealized_pl]));
    alpacaTable("alpaca-orders", data.orders.map(o => [time(o.submitted_at), (o.symbol ?? "—") + " / " + ({buy:"покупка",sell:"продажа"}[o.side] ?? o.side ?? "—"),
      o.qty ?? (o.notional ? o.notional + " USD" : "—"), (o.filled_qty ?? "0") + (o.filled_avg_price ? " @ " + o.filled_avg_price : ""), alpacaOrderStatuses[o.status] ?? o.status]));
  }
  alpacaButtons();
}
async function refreshAlpaca() {
  if (alpacaFetching || alpacaBusy) return;
  alpacaFetching = true;
  try { renderAlpaca(await api("/api/alpaca/state")); }
  catch (error) { $("alpaca-status").textContent = "Нет связи с панелью"; $("alpaca-error").hidden = false; $("alpaca-error").textContent = error.message; }
  finally { alpacaFetching = false; }
}
async function alpacaCommand(path, payload) {
  if (alpacaBusy) return;
  alpacaBusy = true; alpacaCommandError = null; alpacaButtons(); $("alpaca-error").hidden = true;
  try { renderAlpaca(await api(path, payload)); }
  catch (error) { alpacaCommandError = error.message; $("alpaca-error").hidden = false; $("alpaca-error").textContent = error.message; }
  finally { $("alpaca-key").value = ""; $("alpaca-secret").value = ""; alpacaBusy = false; alpacaButtons(); }
}
$("alpaca-connect").addEventListener("submit", event => {event.preventDefault(); alpacaCommand("/api/alpaca/connect", {key:$("alpaca-key").value.trim(),secret:$("alpaca-secret").value.trim()});});
$("alpaca-env").addEventListener("click", () => alpacaCommand("/api/alpaca/connect", {}));
$("alpaca-disconnect").addEventListener("click", () => alpacaCommand("/api/alpaca/disconnect", {}));
refreshAlpaca(); setInterval(refreshAlpaca,2000);
