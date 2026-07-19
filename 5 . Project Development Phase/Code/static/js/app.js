document.addEventListener('DOMContentLoaded', () => {
    // Mobile Hamburger Menu Navigation Toggle
    const hamburgerBtn = document.getElementById('hamburgerBtn');
    const navMenu = document.getElementById('navMenu');

    if (hamburgerBtn && navMenu) {
        hamburgerBtn.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            hamburgerBtn.classList.toggle('active');
            
            // Toggle hamburger icon between bars and X
            const icon = hamburgerBtn.querySelector('i');
            if (icon.classList.contains('fa-bars')) {
                icon.classList.remove('fa-bars');
                icon.classList.add('fa-xmark');
            } else {
                icon.classList.remove('fa-xmark');
                icon.classList.add('fa-bars');
            }
        });
    }

    // Contact Modal Toggle Logic
    const contactBtn = document.getElementById('contactBtn');
    const contactModal = document.getElementById('contactModal');
    const modalCloseBtn = document.getElementById('modalCloseBtn');

    if (contactBtn && contactModal && modalCloseBtn) {
        contactBtn.addEventListener('click', (e) => {
            e.preventDefault();
            contactModal.classList.add('active');
            // Close mobile menu if open
            if (navMenu.classList.contains('active')) {
                navMenu.classList.remove('active');
                hamburgerBtn.classList.remove('active');
                const icon = hamburgerBtn.querySelector('i');
                icon.classList.remove('fa-xmark');
                icon.classList.add('fa-bars');
            }
        });

        modalCloseBtn.addEventListener('click', () => {
            contactModal.classList.remove('active');
        });

        // Close modal when clicking outside of the card
        contactModal.addEventListener('click', (e) => {
            if (e.target === contactModal) {
                contactModal.classList.remove('active');
            }
        });
    }
});
