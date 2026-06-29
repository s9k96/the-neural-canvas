const NAV_PAGES = [
    { id: 'home', label: 'Home', href: 'index.html', home: true },
    { id: 'linear', label: 'Beyond Straight Thinking', href: 'beyond-straight-thinking.html' },
    { id: 'depth-collapse', label: 'Depth Without Creativity', href: 'depth-collapse.html' },
    { id: 'embedding-clustering', label: 'The Company We Keep', href: 'embedding-clustering.html' }
];

function renderSidebarNav() {
    const navList = document.querySelector('.nav-list');
    if (!navList) return;
    navList.innerHTML = '';
    NAV_PAGES.forEach(page => {
        const link = document.createElement('a');
        link.className = `nav-button ${page.home ? 'home-btn' : 'section-btn'}`;
        link.href = page.href;
        link.textContent = page.label;
        if (page.id === window.PAGE_ID) link.classList.add('active');
        navList.appendChild(link);
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

document.addEventListener('DOMContentLoaded', () => {
    renderSidebarNav();
    initSidebarToggle();
});
