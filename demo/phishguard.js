// Port of phishguard/features.py and score_url. Keep the two in sync.
(function (root) {
  "use strict";

  var IPV4 = /^(?:\d{1,3}\.){3}\d{1,3}$/;
  var IPV6 = /^\[?[0-9a-fA-F:]+\]?$/;
  var SCHEME = /^[A-Za-z][A-Za-z0-9+.\-]*:/;

  // Python's str.strip() whitespace set
  var WS = /^[\s\x1c-\x1f\x85]+|[\s\x1c-\x1f\x85]+$/g;

  function strip(s) {
    return s.replace(WS, "");
  }

  // Mirrors urllib.parse.urlparse(...).hostname
  function hostname(url) {
    var raw = strip(url);
    if (raw.indexOf("://") === -1) raw = "http://" + raw;
    raw = raw.replace(/^[\x00-\x20]+/, "").replace(/[\t\r\n]/g, "");
    var m = raw.match(SCHEME);
    var rest = m ? raw.slice(m[0].length) : raw;
    if (rest.slice(0, 2) !== "//") return "";
    rest = rest.slice(2);
    var end = rest.search(/[\/?#]/);
    var netloc = end === -1 ? rest : rest.slice(0, end);
    var hostinfo = netloc.slice(netloc.lastIndexOf("@") + 1);
    var host;
    var br = hostinfo.indexOf("[");
    if (br !== -1) {
      var inner = hostinfo.slice(br + 1);
      var close = inner.indexOf("]");
      host = close === -1 ? inner : inner.slice(0, close);
    } else {
      var colon = hostinfo.indexOf(":");
      host = colon === -1 ? hostinfo : hostinfo.slice(0, colon);
    }
    return host.toLowerCase();
  }

  function withoutScheme(url) {
    var raw = strip(url);
    var i = raw.indexOf("://");
    return i === -1 ? raw : raw.slice(i + 3);
  }

  function extract(url, words) {
    var text = withoutScheme(url);
    var chars = Array.from(text);
    var host = hostname(url);
    var parts = host.split(".").filter(function (p) { return p.length > 0; });
    var ip = host && (IPV4.test(host) || (host.indexOf(":") !== -1 && IPV6.test(host))) ? 1 : 0;
    var lower = text.toLowerCase();
    return {
      url_length: chars.length,
      n_digits: chars.filter(function (c) { return /^\p{Nd}$/u.test(c); }).length,
      n_special: chars.filter(function (c) { return !/^[A-Za-z0-9]$/.test(c); }).length,
      n_subdomains: Math.max(parts.length - 2, 0),
      ip_in_host: ip,
      n_suspicious: words.filter(function (w) { return lower.indexOf(w) !== -1; }).length
    };
  }

  function sigmoid(z) {
    z = Math.min(Math.max(z, -500), 500);
    return 1 / (1 + Math.exp(-z));
  }

  function score(url, model) {
    var feats = extract(url, model.suspicious_words);
    var z = model.bias;
    var rows = model.features.map(function (name, i) {
      var x = (feats[name] - model.mean[i]) / model.std[i];
      var c = model.weights[i] * x;
      z += c;
      return { name: name, raw: feats[name], contribution: c };
    });
    rows.sort(function (a, b) { return Math.abs(b.contribution) - Math.abs(a.contribution); });
    var p = sigmoid(z);
    return {
      probability: p,
      phishing: p >= model.threshold,
      threshold: model.threshold,
      top: rows.slice(0, 5),
      features: feats
    };
  }

  function initPage() {
    var EXAMPLES = [
      "https://example.com",
      "https://github.com/WasimSheikh/phishguard",
      "http://192.168.0.14/secure-login/verify-account",
      "https://update-confirm.example.net/wallet/invoice",
      "paypal.com.account-verify.login.example.ru/signin?id=88213"
    ];

    var form = document.getElementById("form");
    var input = document.getElementById("url");
    var out = document.getElementById("result");
    var model = null;

    function render(url) {
      var r = score(url, model);
      document.getElementById("prob").textContent = r.probability.toFixed(3);
      var v = document.getElementById("verdict");
      v.textContent = (r.phishing ? "phishing" : "likely legit") + " (threshold " + r.threshold.toFixed(2) + ")";
      v.className = r.phishing ? "bad" : "good";
      var body = document.getElementById("top");
      body.innerHTML = "";
      r.top.forEach(function (row) {
        var tr = document.createElement("tr");
        [row.name, (row.contribution >= 0 ? "+" : "") + row.contribution.toFixed(3), String(row.raw)].forEach(function (t) {
          var td = document.createElement("td");
          td.textContent = t;
          tr.appendChild(td);
        });
        body.appendChild(tr);
      });
      out.hidden = false;
    }

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var url = input.value.trim();
      if (url && model) render(url);
    });

    var list = document.getElementById("examples");
    EXAMPLES.forEach(function (u) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = u;
      b.addEventListener("click", function () {
        input.value = u;
        if (model) render(u);
      });
      list.appendChild(b);
    });

    fetch("model.json")
      .then(function (r) { return r.json(); })
      .then(function (m) { model = m; })
      .catch(function () {
        document.getElementById("status").textContent = "Could not load model.json.";
      });
  }

  var api = { extract: extract, score: score, hostname: hostname };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PhishGuard = api;
  if (typeof document !== "undefined") initPage();
})(this);
