const ICONS = {
    home: '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/>',
    linear: '<circle cx="5" cy="19" r="1.6"/><circle cx="19" cy="5" r="1.6"/><path d="M6.4 17.6C10 14 12.5 8.5 17.6 6.4"/>',
    'depth-collapse': '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 13 9 5 9-5"/>',
    'embedding-clustering': '<circle cx="7" cy="8" r="1.5"/><circle cx="6" cy="12.5" r="1.5"/><circle cx="10.5" cy="10.5" r="1.5"/><circle cx="17" cy="16" r="1.5"/><circle cx="19" cy="12" r="1.5"/><circle cx="15" cy="12.5" r="1.5"/>',
    'generalization-gap': '<path d="M3 3v18h18"/><path d="m7 14 4-4 3 3 4-6"/>',
    tokenizer: '<path d="M4 7V5h10v2"/><path d="M9 5v10"/><path d="M7 15h4"/><path d="m14 20 3.5-8 3.5 8"/><path d="M15 17h5"/>',
    'india-first-llm': '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a15 15 0 0 1 0 18 15 15 0 0 1 0-18"/>',
    'data-cleaning': '<path d="M21 4H3l7 8.5V19l4 2v-8.5L21 4Z"/>',
    'dataset-creation': '<path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h10"/><circle cx="18.5" cy="18" r="2.5"/><path d="m20.3 19.8 1.7 1.7"/>',
    'fourier-embeddings': '<path d="M2 12c2-6 4-6 6 0s4 6 6 0 4-6 6 0"/><circle cx="12" cy="12" r="1.4"/>',
    'attention-timeline': '<path d="M3 6h18"/><path d="M3 12h18"/><path d="M3 18h18"/><circle cx="7.5" cy="6" r="1.9"/><circle cx="12" cy="12" r="1.9"/><circle cx="17" cy="18" r="1.9"/>',
    'loss-harness': '<path d="M4 4v15a1 1 0 0 0 1 1h15"/><path d="M7 16c3-1 4.5-8 7-8s2.5 4 5 3"/><circle cx="12" cy="10.5" r="1.6"/>',
    'training-step': '<path d="M3 20h4v-4h5v-4h5V8h4"/><path d="M12 3v6"/><path d="m9.5 6.5 2.5 2.5 2.5-2.5"/>',
    'setting-the-distance': '<path d="M3 4c7 1 7 13 18 15"/><path d="M6 20h9"/><path d="M6 17.5v5M15 17.5v5"/>'
};

const NAV_SECTIONS = [
    { pages: [{ id: 'home', label: 'Home', href: 'index.html', home: true }] },
    {
        label: 'S1 · Network Components',
        pages: [
            { id: 'linear', label: 'Beyond Straight Thinking', href: 's01-understanding-network-components/beyond-straight-thinking.html' },
            { id: 'depth-collapse', label: 'Depth Without Creativity', href: 's01-understanding-network-components/depth-collapse.html' },
            { id: 'embedding-clustering', label: 'The Company We Keep', href: 's01-understanding-network-components/embedding-clustering.html' },
            { id: 'generalization-gap', label: 'Experience Shapes Understanding', href: 's01-understanding-network-components/generalization-gap.html' }
        ]
    },
    { label: 'S2 · Tokenizer', pages: [{ id: 'tokenizer', label: 'One Vocab, Four Languages', href: 's02-tokenizer/tokenizer.html' }] },
    { label: 'S3 · India-First LLM', pages: [{ id: 'india-first-llm', label: 'The World, Viewed From India', href: 's03-india-first-llm/india-first-llm.html' }] },
    { label: 'S4 · Data Cleaning', pages: [{ id: 'data-cleaning', label: 'Raw Data Is Not Training Data', href: 's04-data-cleaning/data-cleaning.html' }] },
    { label: 'S6 · Building the Dataset', pages: [{ id: 'dataset-creation', label: 'Prove What You Trained On', href: 's06-dataset-creation/dataset-creation.html' }] },
    { label: 'S7 · Model Internals', pages: [{ id: 'fourier-embeddings', label: 'Words Made of Waves', href: 's07-model-internals/fourier-embeddings.html' }] },
    { label: 'S8 · Model Architectures', pages: [{ id: 'attention-timeline', label: 'The Field Changes Its Mind', href: 's08-model-architectures/attention-timeline.html' }] },
    { label: 'S9 · Loss Functions', pages: [{ id: 'loss-harness', label: 'Four Ways to Lie About a Loss', href: 's09-loss-functions/loss-harness.html' }] },
    { label: 'S10 · The Training Loop', pages: [{ id: 'training-step', label: 'One Step, and What It Costs', href: 's10-training-loop/training-step.html' }] },
    { label: 'S11 · Optimizers & Schedules', pages: [{ id: 'setting-the-distance', label: 'Setting the Distance', href: 's11-optimizers/setting-the-distance.html' }] }
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

// Home page only: search box + per-session chips over the hand-written topic-card grid.
// Session ids come from each card's data-session; the searchable text is the card's own
// copy plus its NAV_SECTIONS label, so nothing here duplicates the card list.
function initTopicFilter() {
    const grid = document.querySelector('.topic-grid');
    const input = document.getElementById('topicFilter');
    if (!grid || !input) return;

    const labelByHref = {};
    NAV_SECTIONS.forEach(s => s.pages.forEach(p => { labelByHref[p.href] = s.label || ''; }));

    const cards = Array.from(grid.querySelectorAll('.topic-card')).map(el => ({
        el,
        session: el.dataset.session || '',
        label: labelByHref[el.getAttribute('href') || ''] || '',
        text: `${el.textContent} ${labelByHref[el.getAttribute('href') || ''] || ''}`
            .toLowerCase().replace(/\s+/g, ' ')
    }));

    const chipRow = document.getElementById('topicChips');
    const countEl = document.getElementById('topicCount');
    const emptyEl = document.getElementById('topicEmpty');
    const resetEl = document.getElementById('topicReset');
    let session = 'all';

    function apply() {
        const q = input.value.trim().toLowerCase();
        let shown = 0;
        cards.forEach(c => {
            const hit = (session === 'all' || c.session === session) && (!q || c.text.includes(q));
            c.el.classList.toggle('filtered-out', !hit);
            if (hit) shown += 1;
        });
        if (countEl) {
            countEl.textContent = shown === cards.length
                ? `${cards.length} widgets`
                : `${shown} of ${cards.length}`;
        }
        if (emptyEl) emptyEl.hidden = shown > 0;
    }

    function setSession(id) {
        session = id;
        if (!chipRow) return;
        chipRow.querySelectorAll('.topic-chip').forEach(chip => {
            const on = chip.dataset.session === id;
            chip.classList.toggle('active', on);
            chip.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
    }

    if (chipRow) {
        const order = [];
        cards.forEach(c => { if (c.session && !order.includes(c.session)) order.push(c.session); });
        const chips = [{ id: 'all', label: 'All', title: 'Every widget' }].concat(
            order.map(id => ({
                id,
                label: id,
                title: (cards.find(c => c.session === id) || {}).label || id
            }))
        );
        chips.forEach(spec => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'topic-chip';
            chip.dataset.session = spec.id;
            chip.textContent = spec.label;
            chip.title = spec.title;
            // Clicking the active session chip toggles back to All.
            chip.addEventListener('click', () => {
                setSession(session === spec.id && spec.id !== 'all' ? 'all' : spec.id);
                apply();
            });
            chipRow.appendChild(chip);
        });
        setSession('all');
    }

    input.addEventListener('input', apply);
    input.addEventListener('keydown', e => {
        if (e.key !== 'Escape') return;
        input.value = '';
        apply();
        input.blur();
    });

    resetEl?.addEventListener('click', () => {
        input.value = '';
        setSession('all');
        apply();
        input.focus();
    });

    document.addEventListener('keydown', e => {
        if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
        const t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
        e.preventDefault();
        input.focus();
        input.select();
    });

    apply();
}

document.addEventListener('DOMContentLoaded', () => {
    renderSidebarNav();
    initSidebarToggle();
    decorateTopicCards();
    initTopicFilter();
});
