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
  const locales = {
    en: "en_US",
    zh: "zh_CN",
  };
  const socialImageDescriptions = {
    en: "RRE comparison across ORL and Extended YaleB corruption settings",
    zh: "ORL 与 Extended YaleB 各噪声设置下的 RRE 对比",
  };

  const translatableElements = [...document.querySelectorAll("[data-en][data-zh]")];
  const labelledElements = [
    ...document.querySelectorAll("[data-en-aria-label][data-zh-aria-label]"),
  ];
  const imageElements = [...document.querySelectorAll("[data-en-alt][data-zh-alt]")];
  const languageButtons = [...document.querySelectorAll("[data-language]")];
  const description = document.querySelector('meta[name="description"]');
  const socialMetadata = {
    title: [
      document.querySelector('meta[property="og:title"]'),
      document.querySelector('meta[name="twitter:title"]'),
    ],
    description: [
      document.querySelector('meta[property="og:description"]'),
      document.querySelector('meta[name="twitter:description"]'),
    ],
  };
  const openGraphLocale = document.querySelector('meta[property="og:locale"]');
  const socialImageAlternatives = [
    document.querySelector('meta[property="og:image:alt"]'),
    document.querySelector('meta[name="twitter:image:alt"]'),
  ];

  function applyLanguage(language) {
    const nextLanguage = language === "zh" ? "zh" : "en";

    document.documentElement.lang = nextLanguage === "zh" ? "zh-CN" : "en";
    document.title = titles[nextLanguage];
    description?.setAttribute("content", descriptions[nextLanguage]);
    openGraphLocale?.setAttribute("content", locales[nextLanguage]);
    socialMetadata.title.forEach((element) =>
      element?.setAttribute("content", titles[nextLanguage]),
    );
    socialMetadata.description.forEach((element) =>
      element?.setAttribute("content", descriptions[nextLanguage]),
    );
    socialImageAlternatives.forEach((element) =>
      element?.setAttribute("content", socialImageDescriptions[nextLanguage]),
    );

    for (const element of translatableElements) {
      element.textContent = element.dataset[nextLanguage];
    }
    for (const element of labelledElements) {
      const key = nextLanguage === "zh" ? "zhAriaLabel" : "enAriaLabel";
      element.setAttribute("aria-label", element.dataset[key]);
    }
    for (const element of imageElements) {
      const key = nextLanguage === "zh" ? "zhAlt" : "enAlt";
      element.setAttribute("alt", element.dataset[key]);
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

  const documentLanguage = document.documentElement.lang.toLowerCase().startsWith("zh")
    ? "zh"
    : "en";
  let savedLanguage = null;
  try {
    savedLanguage = localStorage.getItem(storageKey);
  } catch {
    savedLanguage = null;
  }

  if (
    (savedLanguage === "en" || savedLanguage === "zh") &&
    savedLanguage !== documentLanguage
  ) {
    window.location.assign(savedLanguage === "zh" ? "zh-CN.html" : "./");
    return;
  }
  applyLanguage(documentLanguage);

  for (const button of languageButtons) {
    button.addEventListener("click", () => {
      const nextLanguage = button.dataset.language === "zh" ? "zh" : "en";
      applyLanguage(nextLanguage);
      window.location.assign(nextLanguage === "zh" ? "zh-CN.html" : "./");
    });
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
