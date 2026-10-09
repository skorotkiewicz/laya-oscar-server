// Run with: node test_floppybird.js. No packages or model downloads.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "floppybird.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page(fetch) {
  const elements = Object.fromEntries(["game", "model", "key", "start", "score", "action", "below", "falling", "latency", "status"]
    .map(id => [id, { value: "", textContent: "", disabled: false, addEventListener() {} }]));
  elements.model.value = "laya";
  elements.model.options = [...html.matchAll(/<option value="([^"]+)">/g)].map(match => ({ value: match[1] }));
  elements.game.getContext = () => new Proxy({}, { get: () => () => {} });
  const context = vm.createContext({ document: { getElementById: id => elements[id] }, fetch,
    AbortController, AbortSignal, performance, requestAnimationFrame: callback => setImmediate(() => callback(0)) });
  vm.runInContext(script, context);
  return { elements, run: code => vm.runInContext(code, context) };
}

async function liveFlight(base) {
  let turns = 0, flaps = 0, waits = 0, capped = false;
  const game = page(async (url, options) => {
    if (url === "/v1/decisions" && turns >= 200) {
      capped = true;
      game.run("stopGame()");
      throw new Error("Live check reached 200 turns.");
    }
    const response = await fetch(new URL(url, base), options);
    if (url === "/v1/decisions" && response.ok) {
      turns++;
      const result = await response.clone().json();
      const below = result.answers.below.noul, falling = result.answers.falling.noul;
      const action = game.run(`chooseAction(${JSON.stringify(below)}, ${JSON.stringify(falling)})`);
      action === "flap" ? flaps++ : waits++;
      if (turns <= 8 || turns % 20 === 0)
        console.log(`Turn ${turns}: ${action}, P(below)=${below}, P(falling)=${falling}, score=${game.run("score")}`);
    }
    return response;
  });
  game.elements.key.value = process.env.API_KEY || "";
  game.run("let seed = 123; Math.random = () => ((seed = seed * 16807 % 2147483647) - 1) / 2147483646;");
  await game.run("startGame()");
  assert.ok(!game.elements.status.textContent.startsWith("Flight stopped:"), game.elements.status.textContent);
  console.log(`Live ${game.elements.model.value}: ${game.run("score")} pipes, ${turns} turns, ${flaps} flaps, ${waits} waits. ` +
    (capped ? "Stopped at the safety limit." : game.elements.status.textContent));
}

async function check() {
  const physics = page();
  for (const [below, falling, expected] of [[.2, .1, "wait"], [.8, .8, "flap"], [.8, .1, "wait"], [.2, .8, "wait"]])
    assert.equal(physics.run(`chooseAction(${below}, ${falling})`), expected);
  for (const invalid of ["null", "undefined", "NaN", '"0.8"', "-0.1", "1.1"])
    assert.throws(() => physics.run(`chooseAction(${invalid}, .8)`), /invalid height\/motion/);
  assert.throws(() => physics.run("chooseAction(.8, undefined)"), /invalid height\/motion/);
  physics.run("pipes[0].top = 120; pipes[0].bottom = 300; bird.y = 300; bird.velocity = -100;");
  assert.match(physics.run("observation().height"), /below the target height/);
  assert.match(physics.run("observation().motion"), /Rising upward/);
  physics.run("reset()");
  physics.run("step(1 / 60)");
  assert.equal(physics.run("bird.velocity"), 15);
  assert.equal(physics.run("bird.y"), 240.25);
  physics.run("bird.velocity = flapVelocity; step(1 / 60)");
  assert.ok(physics.run("bird.y < 240.25"));
  assert.equal(physics.run("reset(); bird.y = floor - radius; step(1 / 60)"), false);
  assert.equal(physics.run("reset(); bird.y = radius; bird.velocity = flapVelocity; step(1 / 60)"), false);
  assert.equal(physics.run("reset(); pipes[0].x = birdX; pipes[0].top = 200; pipes[0].bottom = 380; bird.y = 190; step(1 / 60)"), false);
  assert.equal(physics.run("reset(); pipes[0].x = birdX; pipes[0].top = 200; pipes[0].bottom = 380; bird.y = 290; step(1 / 60)"), true);
  physics.run("reset(); pipes[0].x = birdX - radius - 63; step(1 / 60)");
  assert.equal(physics.run("score"), 1);
  physics.run("pipes[0].x = -100; step(1 / 60)");
  assert.equal(physics.run("pipes.length"), 2);
  assert.ok(Math.abs(physics.run("pipes[1].x - pipes[0].x") - 240) < 1e-9);
  assert.ok(Number.isFinite(physics.run("observation().next_pipe.distance_x")));

  const requests = [];
  const game = page(async (url, options) => {
    requests.push({ url, options });
    return { ok: true, json: async () => url === "/health" ? { models: ["oscar-1"] }
      : { model: "oscar-1", answers: { below: { noul: .2 }, falling: { noul: .8 } } } };
  });
  game.elements.key.value = "demo";
  await game.run("startGame()");
  assert.equal(requests[0].url, "/health");
  assert.ok(requests.length > 1);
  for (const request of requests) {
    assert.equal(request.options.headers.Authorization, "Bearer demo");
    if (request.url !== "/v1/decisions") continue;
    const body = JSON.parse(request.options.body);
    assert.equal(body.model, "oscar-1");
    assert.equal(body.questions.below.type, "noul");
    assert.equal(body.questions.falling.type, "noul");
    assert.ok(Number.isFinite(body.state.bird.velocity_y));
    assert.ok(!("expected" in body));
  }
  assert.match(game.elements.status.textContent, /^Flight over\./);
  assert.equal(game.elements.action.textContent, "WAIT");
  assert.equal(game.elements.below.textContent, "20.0%");
  assert.equal(game.elements.falling.textContent, "80.0%");
  assert.equal(game.elements.start.textContent, "Start flight");
  assert.equal(game.elements.key.disabled, false);

  const hobson = page(async (url, options) => {
    if (url === "/v1/decisions") assert.equal(JSON.parse(options.body).model, "hobson-v21");
    return { ok: true, json: async () => url === "/health" ? { models: ["hobson-v21"] }
      : { answers: { below: { noul: .2 }, falling: { noul: .8 } } } };
  });
  await hobson.run("startGame()");
  assert.equal(hobson.elements.model.value, "hobson-v21");
  assert.equal(hobson.elements.model.options.find(option => option.value === "hobson-v21").disabled, false);
  assert.match(hobson.elements.status.textContent, /^Flight over\./);

  const unauthorized = page(async () => ({ ok: false, status: 401,
    json: async () => ({ detail: "invalid or missing access token" }) }));
  await unauthorized.run("startGame()");
  assert.match(unauthorized.elements.status.textContent, /invalid or missing access token/);
  assert.equal(unauthorized.run("session"), null);

  let complete;
  const stopped = page(() => new Promise(resolve => { complete = resolve; }));
  const flight = stopped.run("startGame()");
  stopped.run("stopGame()");
  complete({ ok: true, json: async () => ({ models: ["laya"] }) });
  await flight;
  assert.equal(stopped.run("score"), 0);
  assert.equal(stopped.run("session"), null);
  assert.match(stopped.elements.status.textContent, /^Stopped\./);
  console.log("Floppybird physics, two-question policy, API requests, auth errors and cancellation passed.");
  if (process.argv[2] === "--live") await liveFlight(process.argv[3] || "http://127.0.0.1:8001");
}

check().catch(error => { console.error(error); process.exitCode = 1; });
