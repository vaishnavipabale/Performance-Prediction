// Mobile nav toggle
document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("navToggle");
  const links = document.querySelector(".nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", () => links.classList.toggle("open"));
  }

  // Score ring entrance animation (re-trigger so CSS transition plays)
  const ring = document.querySelector(".ring-fg");
  if (ring) {
    const target = ring.style.getPropertyValue("--score");
    ring.style.setProperty("--score", 0);
    requestAnimationFrame(() => {
      setTimeout(() => ring.style.setProperty("--score", target), 80);
    });
  }
});
