/* Cloud setup wizard: four guided steps to connect fal.ai, Replicate or Magnific.
   1 Choose a service, 2 Get a key (the provider's own sign-up steps),
   3 Connect (paste and test the key), 4 Done.
   The host's key is tested and saved in the system keychain by the server;
   a guest's key is only kept in the server's memory, for their browser session. */
(function () {
  'use strict';

  const KEY_PATHS = {
    save: (id) => `/api/providers/${encodeURIComponent(id)}/key`,
    session: (id) => `/api/providers/${encodeURIComponent(id)}/session-key`,
  };

  function hostOf(url) {
    try {
      return new URL(url).hostname.replace(/^www\./, '');
    } catch {
      return 'the website';
    }
  }

  function create(deps) {
    const { dialog, api, h, icon, getGuide } = deps;
    const body = dialog.querySelector('#wiz-body');
    const foot = dialog.querySelector('#wiz-foot');
    const stepItems = Array.from(dialog.querySelectorAll('.wiz-steps li'));
    const state = { step: 1, providerId: null, startedAtChoice: true, keyInput: null };

    dialog.addEventListener('close', () => {
      if (dialog.open) return; // it arrives a frame later: opened again meanwhile, so it isn't this visit's
      if (state.keyInput) state.keyInput.value = '';
      state.keyInput = null;
      body.replaceChildren();
      foot.replaceChildren();
    });

    function providers() {
      return deps.providers() || [];
    }

    function provider() {
      return providers().find((p) => p.id === state.providerId) || null;
    }

    function open(options = {}) {
      state.providerId = options.providerId || null;
      state.startedAtChoice = !state.providerId;
      if (!dialog.open) dialog.showModal();
      go(state.providerId ? 2 : 1);
    }

    function go(step) {
      state.step = step;
      stepItems.forEach((item, index) => {
        const n = index + 1;
        item.classList.toggle('is-done', n < step || step === 4);
        item.classList.toggle('is-current', n === step);
        if (n === step) item.setAttribute('aria-current', 'step');
        else item.removeAttribute('aria-current');
      });
      [renderChoose, renderGuide, renderConnect, renderDone][step - 1]();
      dialog.scrollTop = 0;
      // Move focus to the new step's heading, unless the step focused a field itself.
      const heading = body.querySelector('h3');
      if (heading && !body.contains(document.activeElement)) heading.focus();
    }

    function heading(text) {
      return h('h3', { tabindex: '-1' }, text);
    }

    function button(label, { primary = false, onclick, className = '', type = 'button', form } = {}) {
      return h('button', { type, form, class: `btn ${primary ? 'btn-primary' : 'btn-secondary'} ${className}`.trim(), onclick }, label);
    }

    function backButton(step) {
      return h('button', { type: 'button', class: 'btn btn-quiet wiz-back', onclick: () => go(step) }, icon('chevron-left'), 'Back');
    }

    /* Step 1: choose a service */

    function renderChoose() {
      const list = providers();
      const parts = [
        heading('Choose a cloud service'),
        h('p', { class: 'wiz-lede' }, 'Cloud services run bigger AI models on their own computers. You pay them for each photo, usually a few cents, and you only need one.'),
      ];
      if (!list.length) {
        body.replaceChildren(...parts, h('p', null, 'There are no cloud services in this version.'));
        foot.replaceChildren(button('Close', { onclick: () => dialog.close() }));
        return;
      }
      if (!provider()) state.providerId = (list.find((p) => !p.connected) || list[0]).id;
      const group = h('div', { class: 'providers', role: 'radiogroup', 'aria-label': 'Cloud services' });
      for (const p of list) group.append(providerCard(p));
      group.addEventListener('change', (event) => { state.providerId = event.target.value; });
      body.replaceChildren(...parts, group);
      foot.replaceChildren(
        button('Not now', { onclick: () => dialog.close() }),
        button('Next', { primary: true, onclick: () => { state.startedAtChoice = true; go(2); } }));
      for (const p of list) {
        getGuide(p.id).then((guide) => fillProviderCard(group, p.id, guide)).catch(() => fillProviderCard(group, p.id, null));
      }
    }

    function providerCard(p) {
      const fact = (label, key) => h('div', null, h('dt', null, label), h('dd', { 'data-fact': key }, '…'));
      return h('label', { class: 'provider', dataset: { provider: p.id } },
        h('input', { type: 'radio', name: 'wiz-provider', value: p.id, checked: p.id === state.providerId }),
        h('span', { class: 'provider-name' }, p.name, p.connected ? h('span', { class: 'badge badge-soft' }, 'Connected') : null),
        h('span', { class: 'provider-tagline', 'data-fact': 'tagline' }, 'Loading the details…'),
        h('dl', { class: 'provider-facts' }, fact('Best for', 'best_for'), fact('Cost', 'cost_note'), fact('Your photos', 'privacy_note')),
        h('span', { class: 'mode-check', 'aria-hidden': 'true' }, icon('check')));
    }

    function fillProviderCard(group, providerId, guide) {
      const card = Array.from(group.querySelectorAll('.provider')).find((node) => node.dataset.provider === providerId);
      if (!card) return;
      for (const node of card.querySelectorAll('[data-fact]')) {
        const value = guide && guide[node.dataset.fact];
        node.textContent = value || (node.dataset.fact === 'tagline' ? "Details aren't available right now." : 'Not listed');
      }
    }

    /* Step 2: the provider's own sign-up steps */

    function renderGuide() {
      const p = provider();
      if (!p) return go(1);
      const title = heading(`Get your ${p.name} key`);
      body.replaceChildren(title, h('div', { class: 'skeleton' }), h('div', { class: 'skeleton' }));
      foot.replaceChildren(
        state.startedAtChoice ? backButton(1) : button('Not now', { onclick: () => dialog.close(), className: 'wiz-back' }),
        button('I have my key', { primary: true, onclick: () => go(3) }));

      getGuide(p.id).then((guide) => {
        if (state.step !== 2 || state.providerId !== p.id) return;
        const steps = (guide && guide.steps) || [];
        const intro = h('p', { class: 'wiz-lede' }, `Do these on the ${p.name} website, then come back to this window. Each button opens a new tab.`);
        const facts = [];
        if (guide && guide.cost_note) facts.push(h('div', { class: 'alert alert-warn' }, icon('info'), h('p', null, guide.cost_note)));
        const list = h('ol', { class: 'guide-steps' }, steps.map((step) => h('li', { class: 'guide-step' },
          h('h4', null, step.title),
          step.body ? h('p', null, step.body) : null,
          linkButton(step))));
        body.replaceChildren(title, intro, ...facts, list);
      }).catch((err) => {
        if (state.step !== 2) return;
        body.replaceChildren(title, h('div', { class: 'alert alert-danger', role: 'alert' }, icon('alert'), h('p', null, err.message)));
      });
    }

    function linkButton(step) {
      // Only open real https pages from the guide, never anything else.
      if (!step.link_url || !/^https:\/\//i.test(step.link_url)) return null;
      const label = step.link_label || `Open ${hostOf(step.link_url)}`;
      return h('a', { class: 'btn btn-secondary', href: step.link_url, target: '_blank', rel: 'noopener noreferrer' }, label, icon('external'));
    }

    /* Step 3: paste and test the key */

    function renderConnect() {
      const p = provider();
      if (!p) return go(1);
      const host = deps.isHost();
      const input = h('input', {
        class: 'input', id: 'wiz-key', type: 'password', autocomplete: 'off', autocapitalize: 'off',
        spellcheck: 'false', 'aria-describedby': 'wiz-key-hint wiz-key-note',
      });
      state.keyInput = input;
      const toggle = h('button', { type: 'button', class: 'btn btn-quiet btn-sm key-toggle', 'aria-pressed': 'false', 'aria-controls': 'wiz-key' }, icon('eye'), 'Show');
      toggle.addEventListener('click', () => {
        const show = input.type === 'password';
        input.type = show ? 'text' : 'password';
        toggle.setAttribute('aria-pressed', String(show));
        toggle.replaceChildren(icon(show ? 'eye-off' : 'eye'), show ? 'Hide' : 'Show');
        input.focus();
      });
      const hint = h('p', { class: 'hint', id: 'wiz-key-hint' });
      const error = h('p', { class: 'field-error', id: 'wiz-key-error', role: 'alert', hidden: true });
      const note = h('p', { class: 'hint', id: 'wiz-key-note' }, host
        ? `If it works, the key is saved in this computer's keychain. It is only ever sent to ${p.name}.`
        : `Re:Solution keeps your key in its memory on the main computer, never on disk, and only sends it to ${p.name}. It's forgotten when you close your browser, after a few hours without use, or when you click "Forget my key".`);

      const form = h('form', { class: 'key-field', id: 'wiz-key-form', novalidate: true },
        h('label', { class: 'field-label', for: 'wiz-key' }, `Paste your ${p.name} key`),
        h('div', { class: 'key-row' }, input, toggle),
        hint, error, note);
      body.replaceChildren(heading(`Connect ${p.name}`), h('p', { class: 'wiz-lede' }, 'Copy the key from their website and paste it here. Testing it costs nothing.'), form);

      const submitLabel = host ? 'Test connection' : 'Use my own key for this visit';
      const submit = h('button', { type: 'submit', form: 'wiz-key-form', class: 'btn btn-primary' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), h('span', { class: 'btn-label' }, submitLabel));
      foot.replaceChildren(backButton(2), submit);

      getGuide(p.id).then((guide) => { if (guide && guide.key_hint) hint.textContent = guide.key_hint; }).catch(() => {});

      const showError = (message) => {
        error.textContent = message;
        error.hidden = false;
        input.setAttribute('aria-invalid', 'true');
        input.setAttribute('aria-describedby', 'wiz-key-error wiz-key-hint wiz-key-note');
      };

      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const key = input.value.trim();
        if (!key) {
          showError('Paste your key first.');
          input.focus();
          return;
        }
        error.hidden = true;
        input.removeAttribute('aria-invalid');
        submit.disabled = true;
        submit.classList.add('is-busy');
        submit.querySelector('.btn-label').textContent = 'Testing…';
        try {
          const result = await api(host ? KEY_PATHS.save(p.id) : KEY_PATHS.session(p.id), { method: 'POST', body: { key } });
          if (result && result.ok === false) throw new Error(result.message || result.detail || `${p.name} didn't accept that key.`);
          input.value = '';
          deps.onConnected(p.id);
          go(4);
        } catch (err) {
          showError(err.message || `${p.name} didn't accept that key. Check that you copied all of it.`);
          input.focus();
          input.select();
        } finally {
          submit.disabled = false;
          submit.classList.remove('is-busy');
          const label = submit.querySelector('.btn-label');
          if (label) label.textContent = submitLabel;
        }
      });
      input.focus();
    }

    /* Step 4: done */

    function renderDone() {
      const p = provider();
      if (!p) return go(1);
      const host = deps.isHost();
      body.replaceChildren(h('div', { class: 'success' },
        h('span', { class: 'success-mark', 'aria-hidden': 'true' }, icon('check')),
        heading(`${p.name} is connected`),
        h('p', { class: 'wiz-lede' }, host
          ? "The key is saved in this computer's keychain. You can remove it any time in Settings."
          : 'You can use it in this browser until you close it. On a shared phone or computer, click "Forget my key" under the qualities as soon as you\'re done.'),
        h('p', { class: 'wiz-lede' }, 'Its qualities now show up under "Choose the quality" when you upscale photos.')));
      foot.replaceChildren(
        button('Close', { onclick: () => dialog.close() }),
        button('Try it on an image', { primary: true, onclick: () => { dialog.close(); deps.onTryIt(p.id); } }));
    }

    return { open };
  }

  window.UpscalerWizard = { create };
})();
