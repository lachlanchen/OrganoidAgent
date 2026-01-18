self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open("organoid-agent-v1").then((cache) =>
      cache.addAll([
        "/",
        "/static/index.html",
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
  event.respondWith(
    caches.match(request).then((cached) => {
      return (
        cached ||
        fetch(request).then((response) => {
          const copy = response.clone();
          caches.open("organoid-agent-v1").then((cache) => {
            cache.put(request, copy);
          });
          return response;
        })
      );
    })
  );
});
