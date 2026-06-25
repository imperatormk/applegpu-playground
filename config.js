window.PLAYGROUND_API =
  location.hostname === "localhost" || location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://applegpu-playground.duckdns.org";
