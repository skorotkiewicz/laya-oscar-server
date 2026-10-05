// Run with: node test_floppybird.js. No packages or model downloads.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "floppybird.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page(fetch) {
  const elements = Object.fromEntries(["game", "model", "key", "start", "score", "action", "probability", "latency", "status"]
    .map(id => [id, { value: "", textContent: "", disabled: false, addEventListener() {} }]));
  elements.model.value = "laya";
  elements.model.options = [{ value: "laya" }, { value: "oscar-1" }];
  elements.game.getContext = () => new Proxy({}, { get: () => () => {} });
  const context = vm.createContext({ document: { getElementById: id => elements[id] }, fetch,
    AbortController, AbortSignal, performance, requestAnimationFrame: callback => setImmediate(() => callback(0)) });
  vm.runInContext(script, context);
  return { elements, run: code => vm.runInContext(code, context) };
}

async function check() {
  const physics = page();
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
      : { model: "oscar-1", answers: { move: { choice: "wait", probabilities: { wait: .8, flap: .2 } } } } };
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
    assert.equal(body.questions.move.type, "choice");
    assert.deepEqual(Object.keys(body.questions.move.criteria), ["flap", "wait"]);
    assert.ok(Number.isFinite(body.state.bird.velocity_y));
    assert.ok(!("expected" in body));
  }
  assert.match(game.elements.status.textContent, /^Flight over\./);
  assert.equal(game.elements.action.textContent, "WAIT");
  assert.equal(game.elements.probability.textContent, "80.0%");
  assert.equal(game.elements.start.textContent, "Start flight");
  assert.equal(game.elements.key.disabled, false);

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
  console.log("Floppybird physics, API requests, auth errors and cancellation passed.");
}

check().catch(error => { console.error(error); process.exitCode = 1; });
