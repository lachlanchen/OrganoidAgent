self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key !== "organoid-agent-v2").map((key) => caches.delete(key)))
    )
  );
});

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open("organoid-agent-v2").then((cache) =>
      cache.addAll([
        "/",
        "/static/styles.css",
        "/static/app.js",
        "/static/manifest.json",
        "/static/icons/icon-192.png",
        "/static/icons/icon-512.png",
      ])
    )
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.pathname.startsWith("/api/")) return;
  event.respondWith(
    caches.match(request).then((cached) => {
      return (
        cached ||
        fetch(request).then((response) => {
          const copy = response.clone();
          caches.open("organoid-agent-v2").then((cache) => {
            cache.put(request, copy);
          });
          return response;
        })
      );
    })
  );
});
