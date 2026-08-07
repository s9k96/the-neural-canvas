const ICONS = {
    home: '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/>',
    linear: '<circle cx="5" cy="19" r="1.6"/><circle cx="19" cy="5" r="1.6"/><path d="M6.4 17.6C10 14 12.5 8.5 17.6 6.4"/>',
    'depth-collapse': '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 13 9 5 9-5"/>',
    'embedding-clustering': '<circle cx="7" cy="8" r="1.5"/><circle cx="6" cy="12.5" r="1.5"/><circle cx="10.5" cy="10.5" r="1.5"/><circle cx="17" cy="16" r="1.5"/><circle cx="19" cy="12" r="1.5"/><circle cx="15" cy="12.5" r="1.5"/>',
    'generalization-gap': '<path d="M3 3v18h18"/><path d="m7 14 4-4 3 3 4-6"/>',
    tokenizer: '<path d="M4 7V5h10v2"/><path d="M9 5v10"/><path d="M7 15h4"/><path d="m14 20 3.5-8 3.5 8"/><path d="M15 17h5"/>',
    'india-first-llm': '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a15 15 0 0 1 0 18 15 15 0 0 1 0-18"/>',
    'data-cleaning': '<path d="M21 4H3l7 8.5V19l4 2v-8.5L21 4Z"/>',
    'dataset-creation': '<path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h10"/><circle cx="18.5" cy="18" r="2.5"/><path d="m20.3 19.8 1.7 1.7"/>'
};

const NAV_SECTIONS = [
    { pages: [{ id: 'home', label: 'Home', href: 'index.html', home: true }] },
    {
        label: 'S1 · Network Components',
        pages: [
            { id: 'linear', label: 'Beyond Straight Thinking', href: 'S1-understanding-network-components/beyond-straight-thinking.html' },
            { id: 'depth-collapse', label: 'Depth Without Creativity', href: 'S1-understanding-network-components/depth-collapse.html' },
            { id: 'embedding-clustering', label: 'The Company We Keep', href: 'S1-understanding-network-components/embedding-clustering.html' },
            { id: 'generalization-gap', label: 'Experience Shapes Understanding', href: 'S1-understanding-network-components/generalization-gap.html' }
        ]
    },
    { label: 'S2 · Tokenizer', pages: [{ id: 'tokenizer', label: 'One Vocab, Four Languages', href: 's2-tokenizer/tokenizer.html' }] },
    { label: 'S3 · India-First LLM', pages: [{ id: 'india-first-llm', label: 'The World, Viewed From India', href: 's3-india-first-llm/india-first-llm.html' }] },
    { label: 'S4 · Data Cleaning', pages: [{ id: 'data-cleaning', label: 'Raw Data Is Not Training Data', href: 's4-data-cleaning/data-cleaning.html' }] },
    { label: 'S6 · Building the Dataset', pages: [{ id: 'dataset-creation', label: 'Prove What You Trained On', href: 's6-dataset-creation/dataset-creation.html' }] }
];

function iconSvg(id) {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[id] || ''}</svg>`;
}

function renderSidebarHeader(rootPath) {
    const header = document.querySelector('.sidebar-header');
    if (!header) return;
    header.innerHTML =
        `<a class="sidebar-brand" href="${rootPath}index.html" aria-label="Neural Canvas home">
            <span class="brand-mark">
                <svg viewBox="0 0 32 32" aria-hidden="true">
                    <g stroke="var(--accent)" stroke-width="1.6" stroke-linecap="round">
                        <line x1="8" y1="9" x2="16" y2="16"/><line x1="8" y1="23" x2="16" y2="16"/>
                        <line x1="16" y1="16" x2="24" y2="9"/><line x1="16" y1="16" x2="24" y2="16"/><line x1="16" y1="16" x2="24" y2="23"/>
                    </g>
                    <g fill="currentColor"><circle cx="8" cy="9" r="2.3"/><circle cx="8" cy="23" r="2.3"/><circle cx="24" cy="9" r="2.3"/><circle cx="24" cy="16" r="2.3"/><circle cx="24" cy="23" r="2.3"/></g>
                    <circle cx="16" cy="16" r="3" fill="var(--accent)"/>
                </svg>
            </span>
            <span class="brand-text">Neural Canvas<small>ML, made visible</small></span>
        </a>
        <button class="sidebar-close" id="sidebarClose" aria-label="Close navigation">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/></svg>
        </button>`;
}

function renderSidebarNav() {
    const navList = document.querySelector('.nav-list');
    if (!navList) return;
    const rootPath = window.ROOT_PATH || '';
    renderSidebarHeader(rootPath);
    navList.innerHTML = '';
    NAV_SECTIONS.forEach(section => {
        if (section.label) {
            const label = document.createElement('div');
            label.className = 'nav-section-label';
            label.textContent = section.label;
            navList.appendChild(label);
        }
        section.pages.forEach(page => {
            const link = document.createElement('a');
            link.className = `nav-button ${page.home ? 'home-btn' : 'section-btn'}`;
            link.href = rootPath + page.href;
            link.innerHTML = `<span class="nav-ico">${iconSvg(page.id)}</span><span class="nav-label">${page.label}</span>`;
            if (page.id === window.PAGE_ID) {
                link.classList.add('active');
                link.setAttribute('aria-current', 'page');
            }
            navList.appendChild(link);
        });
    });
}

function initSidebarToggle() {
    const toggle = document.getElementById('sidebarToggle');
    const close = document.getElementById('sidebarClose');
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebarBackdrop');
    if (!toggle || !sidebar || !backdrop) return;

    function openNav() {
        sidebar.classList.add('open');
        backdrop.classList.add('visible');
    }

    function closeNav() {
        sidebar.classList.remove('open');
        backdrop.classList.remove('visible');
    }

    toggle.addEventListener('click', openNav);
    close?.addEventListener('click', closeNav);
    backdrop.addEventListener('click', closeNav);
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeNav();
    });
    sidebar.querySelectorAll('.nav-button').forEach(link => link.addEventListener('click', closeNav));
}

function decorateTopicCards() {
    const byHref = {};
    NAV_SECTIONS.forEach(s => s.pages.forEach(p => { byHref[p.href] = p.id; }));
    document.querySelectorAll('.topic-card').forEach(card => {
        const id = byHref[card.getAttribute('href') || ''];
        if (!ICONS[id] || card.querySelector('.card-ico')) return;
        const ico = document.createElement('span');
        ico.className = 'card-ico';
        ico.innerHTML = iconSvg(id);
        card.insertBefore(ico, card.firstChild);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    renderSidebarNav();
    initSidebarToggle();
    decorateTopicCards();
});
