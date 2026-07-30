(() => {
  const storageKey = "robust-nmf-language";
  const descriptions = {
    en: "A bilingual reconstruction dossier for a four-person robust NMF face reconstruction project.",
    zh: "一份双语重建研究档案，记录四人团队完成的鲁棒 NMF 人脸重建项目。",
  };
  const titles = {
    en: "Robust NMF — Reconstruction Dossier",
    zh: "鲁棒 NMF — 重建研究档案",
  };

  const translatableElements = [...document.querySelectorAll("[data-en][data-zh]")];
  const languageButtons = [...document.querySelectorAll("[data-language]")];
  const description = document.querySelector('meta[name="description"]');

  function applyLanguage(language) {
    const nextLanguage = language === "zh" ? "zh" : "en";

    document.documentElement.lang = nextLanguage === "zh" ? "zh-CN" : "en";
    document.title = titles[nextLanguage];
    description?.setAttribute("content", descriptions[nextLanguage]);

    for (const element of translatableElements) {
      element.textContent = element.dataset[nextLanguage];
    }

    for (const button of languageButtons) {
      button.setAttribute("aria-pressed", String(button.dataset.language === nextLanguage));
    }

    try {
      localStorage.setItem(storageKey, nextLanguage);
    } catch {
      // Storage can be unavailable in privacy-restricted browsing contexts.
    }
  }

  let savedLanguage = "en";
  try {
    savedLanguage = localStorage.getItem(storageKey) || "en";
  } catch {
    savedLanguage = "en";
  }

  applyLanguage(savedLanguage);

  for (const button of languageButtons) {
    button.addEventListener("click", () => applyLanguage(button.dataset.language));
  }

  const revealTargets = document.querySelectorAll(
    ".dossier-section > :not(.section-label), .result-figure, .closing > *",
  );
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if ("IntersectionObserver" in window && !reduceMotion) {
    revealTargets.forEach((element) => element.setAttribute("data-reveal", ""));
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        }
      },
      { threshold: 0.12 },
    );
    revealTargets.forEach((element) => observer.observe(element));
  }
})();
