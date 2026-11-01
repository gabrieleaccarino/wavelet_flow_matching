// Tabs
document.querySelectorAll('.tab').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var id = btn.getAttribute('data-tab');
    document.querySelectorAll('.tab').forEach(function (b) { b.classList.toggle('active', b === btn); });
    document.querySelectorAll('.tab-panel').forEach(function (p) {
      var on = p.id === 'tab-' + id;
      p.classList.toggle('active', on);
    });
  });
});

// Fall back to the static figure (with a "coming soon" label) when a GIF is missing
document.querySelectorAll('[data-gif]').forEach(function (frame) {
  var img = frame.querySelector('img');
  var fallback = function () {
    if (img.dataset.failed) return;
    img.dataset.failed = '1';
    frame.classList.add('missing');
    img.src = img.getAttribute('data-fallback');
  };
  img.addEventListener('error', fallback);
  if (img.complete && img.naturalWidth === 0) fallback();
});

// Copy BibTeX
var copyBtn = document.querySelector('.copy-btn');
if (copyBtn) {
  copyBtn.addEventListener('click', function () {
    var text = document.getElementById('bibtex-code').innerText;
    navigator.clipboard.writeText(text).then(function () {
      copyBtn.textContent = 'Copied!';
      setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1500);
    });
  });
}
