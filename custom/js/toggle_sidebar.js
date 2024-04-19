(function() {
    var toggleableHeaders = document.querySelectorAll('.md-sidebar-nav .md-toggle');
    toggleableHeaders.forEach(function(header) {
      header.addEventListener('click', function() {
        this.parentElement.classList.toggle('md-nav--toggle');
      });
    });
  })();