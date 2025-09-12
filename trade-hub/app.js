(() => {
	const STATE = {
		products: [],
		filteredProducts: [],
		categories: new Set(),
		page: 1,
		pageSize: 8,
		cart: loadJSON('th_cart', []),
		wishlist: loadJSON('th_wishlist', []),
		theme: localStorage.getItem('th_theme') || 'dark',
		dealIndex: 0,
	};

	const els = {
		body: document.body,
		menuToggle: document.querySelector('.menu-toggle'),
		nav: document.getElementById('primary-nav'),
		links: document.querySelectorAll('[data-route]'),
		panels: document.querySelectorAll('[data-route-panel]'),
		year: document.getElementById('year'),
		sortSelect: document.getElementById('sortSelect'),
		categoryFilter: document.getElementById('categoryFilter'),
		searchInput: document.getElementById('searchInput'),
		productGrid: document.getElementById('productGrid'),
		prevPage: document.getElementById('prevPage'),
		nextPage: document.getElementById('nextPage'),
		pageInfo: document.getElementById('pageInfo'),
		cartCount: document.getElementById('cartCount'),
		wishlistCount: document.getElementById('wishlistCount'),
		modals: document.querySelectorAll('.modal'),
		dealsTrack: document.getElementById('dealsTrack'),
		themeToggle: document.getElementById('themeToggle'),
	};

	init();

	function init() {
		els.year.textContent = new Date().getFullYear();
		applyTheme(STATE.theme);
		setupRouting();
		setupHeader();
		setupModals();
		setupCatalogControls();
		loadProducts();
		updateBadges();
		setupDealsCarousel();
	}

	function setupHeader() {
		els.menuToggle?.addEventListener('click', () => {
			const open = !els.nav.classList.contains('open');
			els.nav.classList.toggle('open', open);
			els.menuToggle.setAttribute('aria-expanded', String(open));
		});

		els.themeToggle?.addEventListener('click', () => {
			STATE.theme = STATE.theme === 'dark' ? 'light' : 'dark';
			localStorage.setItem('th_theme', STATE.theme);
			applyTheme(STATE.theme);
			els.themeToggle.setAttribute('aria-pressed', String(STATE.theme === 'light'));
		});
	}

	function applyTheme(theme) {
		if (theme === 'light') {
			document.documentElement.classList.add('light');
		} else {
			document.documentElement.classList.remove('light');
		}
	}

	function setupRouting() {
		function navigate(hash) {
			const target = (hash || '#home').replace('#', '');
			els.links.forEach(a => a.setAttribute('aria-current', a.getAttribute('href') === '#' + target ? 'page' : 'false'));
			els.panels.forEach(p => p.classList.toggle('hidden', p.id !== target));
			if (els.nav.classList.contains('open')) {
				els.nav.classList.remove('open');
				els.menuToggle?.setAttribute('aria-expanded', 'false');
			}
			if (target === 'catalog') {
				renderCatalog();
			}
			if (target === 'home') {
				// focus main for accessibility
				document.getElementById('main')?.focus();
			}
		}
		window.addEventListener('hashchange', () => navigate(location.hash));
		navigate(location.hash);
	}

	function setupModals() {
		document.addEventListener('click', (e) => {
			const openTarget = e.target.closest('[data-open]');
			if (openTarget) {
				const id = openTarget.getAttribute('data-open');
				openModal(id);
			}
			const closeTarget = e.target.closest('[data-close]');
			if (closeTarget) closeModal(closeTarget.closest('.modal').id);
		});
		document.addEventListener('keydown', (e) => {
			if (e.key === 'Escape') {
				els.modals.forEach(m => m.getAttribute('aria-hidden') === 'false' && closeModal(m.id));
			}
		});
	}

	function openModal(id) {
		const m = document.getElementById(id);
		if (!m) return;
		m.setAttribute('aria-hidden', 'false');
		trapFocus(m);
		if (id === 'cart-modal') renderCart();
		if (id === 'wishlist-modal') renderWishlist();
	}
	function closeModal(id) {
		const m = document.getElementById(id);
		if (!m) return;
		m.setAttribute('aria-hidden', 'true');
		releaseFocus();
	}

	let lastFocus = null;
	function trapFocus(container) {
		lastFocus = document.activeElement;
		const focusables = container.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
		const first = focusables[0];
		const last = focusables[focusables.length - 1];
		container.addEventListener('keydown', handle);
		function handle(e) {
			if (e.key !== 'Tab') return;
			if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
			else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
		}
		setTimeout(() => first?.focus(), 0);
	}
	function releaseFocus() { lastFocus?.focus?.(); }

	function setupCatalogControls() {
		['change', 'input'].forEach(evt => {
			els.sortSelect?.addEventListener(evt, handleFilterSortSearch);
			els.categoryFilter?.addEventListener(evt, handleFilterSortSearch);
			els.searchInput?.addEventListener(evt, debounce(handleFilterSortSearch, 200));
		});
		els.prevPage?.addEventListener('click', () => { if (STATE.page > 1) { STATE.page--; renderCatalog(); }});
		els.nextPage?.addEventListener('click', () => { STATE.page++; renderCatalog(); });
	}

	function handleFilterSortSearch() {
		STATE.page = 1;
		applyFilters();
		renderCatalog();
	}

	function loadProducts() {
		// Seed demo products (would be fetched in a real app)
		const demo = Array.from({ length: 36 }).map((_, i) => {
			const category = ['Electronics', 'Home', 'Fashion', 'Outdoors'][i % 4];
			const price = Math.round((Math.sin(i) + 2) * 25) + (i % 5) * 9;
			const rating = (Math.round(((Math.cos(i) + 1) * 2.5) + 2) * 10) / 10;
			return {
				id: 'p' + (i + 1),
				title: `${category} Item ${i + 1}`,
				category,
				price,
				rating,
				image: `https://picsum.photos/seed/tradehub-${i}/480/320`,
				popularity: Math.floor(Math.random() * 1000),
			};
		});
		STATE.products = demo;
		STATE.categories = new Set(demo.map(p => p.category));
		els.categoryFilter.innerHTML = ['<option value="">All categories</option>', ...Array.from(STATE.categories).map(c => `<option value="${c}">${c}</option>`)].join('');
		applyFilters();
		renderDeals();
	}

	function applyFilters() {
		const term = (els.searchInput?.value || '').toLowerCase().trim();
		const cat = els.categoryFilter?.value || '';
		const sort = els.sortSelect?.value || 'popularity';
		let list = STATE.products.filter(p => (!cat || p.category === cat) && (!term || p.title.toLowerCase().includes(term)));
		if (sort === 'price-asc') list.sort((a,b) => a.price - b.price);
		else if (sort === 'price-desc') list.sort((a,b) => b.price - a.price);
		else if (sort === 'rating') list.sort((a,b) => b.rating - a.rating);
		else list.sort((a,b) => b.popularity - a.popularity);
		STATE.filteredProducts = list;
	}

	function renderCatalog() {
		const start = (STATE.page - 1) * STATE.pageSize;
		const items = STATE.filteredProducts.slice(start, start + STATE.pageSize);
		els.productGrid.innerHTML = items.map(renderProductCard).join('');
		els.prevPage.disabled = STATE.page === 1;
		const totalPages = Math.max(1, Math.ceil(STATE.filteredProducts.length / STATE.pageSize));
		els.nextPage.disabled = STATE.page >= totalPages;
		els.pageInfo.textContent = `Page ${STATE.page} of ${totalPages}`;
		els.productGrid.querySelectorAll('[data-add-cart]').forEach(btn => btn.addEventListener('click', () => addToCart(btn.dataset.id)));
		els.productGrid.querySelectorAll('[data-add-wishlist]').forEach(btn => btn.addEventListener('click', () => toggleWishlist(btn.dataset.id)));
	}

	function renderProductCard(p) {
		const wished = STATE.wishlist.includes(p.id);
		return `
			<article class="card" aria-label="${escapeHtml(p.title)}">
				<img loading="lazy" src="${p.image}" alt="${escapeHtml(p.title)}">
				<div class="card-body">
					<h3 class="card-title">${escapeHtml(p.title)}</h3>
					<div class="card-meta">${escapeHtml(p.category)} • ⭐ ${p.rating}</div>
					<div class="card-price">$${p.price}</div>
					<div class="card-actions">
						<button class="button button-small" data-add-cart data-id="${p.id}">Add to cart</button>
						<button class="icon-button" aria-pressed="${wished}" data-add-wishlist data-id="${p.id}">${wished ? '♥' : '♡'}</button>
					</div>
				</div>
			</article>
		`;
	}

	function addToCart(id) {
		const product = STATE.products.find(p => p.id === id);
		if (!product) return;
		const existing = STATE.cart.find(i => i.id === id);
		if (existing) existing.qty += 1; else STATE.cart.push({ id, qty: 1 });
		saveJSON('th_cart', STATE.cart);
		updateBadges();
	}

	function toggleWishlist(id) {
		const index = STATE.wishlist.indexOf(id);
		if (index >= 0) STATE.wishlist.splice(index, 1); else STATE.wishlist.push(id);
		saveJSON('th_wishlist', STATE.wishlist);
		updateBadges();
		renderCatalog();
	}

	function renderCart() {
		const list = STATE.cart.map(item => {
			const p = STATE.products.find(p => p.id === item.id);
			return `<div class="row" data-id="${p.id}">
				<div>${escapeHtml(p.title)}</div>
				<div>Qty: <button class="icon-button" data-dec>-</button> ${item.qty} <button class="icon-button" data-inc>+</button></div>
				<div>$${p.price * item.qty}</div>
				<button class="icon-button" data-remove aria-label="Remove">✕</button>
			</div>`;
		}).join('');
		const total = STATE.cart.reduce((sum, it) => sum + (STATE.products.find(p => p.id === it.id)?.price || 0) * it.qty, 0);
		document.getElementById('cartItems').innerHTML = `<div class="cart-list">${list || '<p>Your cart is empty.</p>'}</div><div class="cart-total"><strong>Total:</strong> $${total}</div>`;
		document.getElementById('cartItems').querySelectorAll('[data-inc]').forEach(b => b.addEventListener('click', () => changeQty(b.closest('.row').dataset.id, 1)));
		document.getElementById('cartItems').querySelectorAll('[data-dec]').forEach(b => b.addEventListener('click', () => changeQty(b.closest('.row').dataset.id, -1)));
		document.getElementById('cartItems').querySelectorAll('[data-remove]').forEach(b => b.addEventListener('click', () => removeFromCart(b.closest('.row').dataset.id)));
		document.getElementById('checkoutButton')?.addEventListener('click', fakeCheckout);
	}

	function changeQty(id, delta) {
		const it = STATE.cart.find(i => i.id === id);
		if (!it) return;
		it.qty = Math.max(0, it.qty + delta);
		if (it.qty === 0) STATE.cart = STATE.cart.filter(i => i.id !== id);
		saveJSON('th_cart', STATE.cart);
		renderCart();
		updateBadges();
	}
	function removeFromCart(id) { STATE.cart = STATE.cart.filter(i => i.id !== id); saveJSON('th_cart', STATE.cart); renderCart(); updateBadges(); }

	function renderWishlist() {
		const items = STATE.wishlist.map(id => STATE.products.find(p => p.id === id)).filter(Boolean);
		document.getElementById('wishlistItems').innerHTML = items.map(p => `
			<div class="row" data-id="${p.id}">
				<div>${escapeHtml(p.title)}</div>
				<div class="card-actions">
					<button class="button button-small" data-add-cart="${p.id}">Add to cart</button>
					<button class="icon-button" data-remove>Remove</button>
				</div>
			</div>
		`).join('') || '<p>Your wishlist is empty.</p>';
		document.getElementById('wishlistItems').querySelectorAll('[data-add-cart]').forEach(b => b.addEventListener('click', () => { addToCart(b.getAttribute('data-add-cart')); }));
		document.getElementById('wishlistItems').querySelectorAll('[data-remove]').forEach(b => b.addEventListener('click', () => { toggleWishlist(b.closest('.row').dataset.id); renderWishlist(); }));
	}

	function fakeCheckout() {
		openModal('auth-modal');
	}

	// Deals carousel
	function setupDealsCarousel() {
		document.querySelector('[data-carousel="prev"]').addEventListener('click', () => shiftDeal(-1));
		document.querySelector('[data-carousel="next"]').addEventListener('click', () => shiftDeal(1));
	}
	function renderDeals() {
		const picks = STATE.products.slice(0, 10);
		els.dealsTrack.innerHTML = picks.map(p => `<div class="carousel-item"><strong>${escapeHtml(p.title)}</strong><div style="margin-top:6px" class="card-price">Now $${Math.max(1, Math.round(p.price * 0.85))}</div></div>`).join('');
	}
	function shiftDeal(dir) {
		STATE.dealIndex = (STATE.dealIndex + dir + 10) % 10;
		els.dealsTrack.scrollTo({ left: STATE.dealIndex * els.dealsTrack.clientWidth * 0.85, behavior: 'smooth' });
	}

	// Auth form
	const authForm = document.getElementById('authForm');
	if (authForm) {
		authForm.addEventListener('submit', (e) => {
			e.preventDefault();
			const email = document.getElementById('authEmail');
			const pass = document.getElementById('authPassword');
			const hint = document.getElementById('authHint');
			if (!email.validity.valid || !pass.validity.valid) {
				hint.textContent = 'Enter a valid email and a 6+ char password.';
				return;
			}
			hint.textContent = 'Signed in (demo). You can now checkout.';
			closeModal('auth-modal');
			openModal('cart-modal');
		});
	}

	// Utils
	function updateBadges() {
		const cartQty = STATE.cart.reduce((n, i) => n + i.qty, 0);
		els.cartCount.textContent = String(cartQty);
		els.wishlistCount.textContent = String(STATE.wishlist.length);
	}
	function debounce(fn, wait) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), wait); }; }
	function saveJSON(k, v) { localStorage.setItem(k, JSON.stringify(v)); }
	function loadJSON(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } }
	function escapeHtml(s) { return s.replace(/[&<>"]+/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
})();

