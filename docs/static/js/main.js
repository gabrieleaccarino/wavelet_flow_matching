// Tabs
document.querySelectorAll('.tab').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var id = btn.getAttribute('data-tab');
    document.querySelectorAll('.tab').forEach(function (b) { b.classList.toggle('active', b === btn); });
    document.querySelectorAll('.tab-panel').forEach(function (p) {
      var on = p.id === 'tab-' + id;
      p.classList.toggle('active', on);
      if (!on) p.querySelectorAll('video').forEach(function (v) { v.pause(); });
    });
  });
});

// Show a "coming soon" label (and keep the poster) when a video file is missing
document.querySelectorAll('[data-video]').forEach(function (frame) {
  var video = frame.querySelector('video');
  var source = frame.querySelector('source');
  var mark = function () { frame.classList.add('missing'); video.removeAttribute('controls'); };
  if (source) source.addEventListener('error', mark);
  video.addEventListener('error', mark);
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
