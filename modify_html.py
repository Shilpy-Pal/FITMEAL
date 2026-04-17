import os
import re

TEMPLATE_DIR = r"c:\Users\user\Desktop\html 2024\FITMEAL_CLEAN\templates"

new_nav_button = """          {% if session.get('user') %}
            <a href="/logout" class="btn btn-light text-danger fw-bold px-3 py-1 rounded-pill shadow-sm">Logout</a>
          {% else %}
            <a href="/" class="btn btn-light text-success fw-bold px-3 py-1 rounded-pill shadow-sm">Sign Up</a>
          {% endif %}"""

new_footer = """  <footer class="text-center text-lg-start pt-5 pb-3 mt-5" style="background-color: #2c3e50; color: #ecf0f1; font-family: 'Poppins', sans-serif;">
    <div class="container text-center text-md-start">
      <div class="row">
        <div class="col-md-3 col-lg-4 col-xl-3 mx-auto mb-4">
          <h6 class="text-uppercase fw-bold mb-4" style="color: #00b09b;">
            FitMeal
          </h6>
          <p style="font-size: 0.95rem; opacity: 0.85;">Your ultimate destination for balanced nutrition, fitness recipes, and dynamic meal tracking. Fuel your body, fuel your life.</p>
        </div>
        <div class="col-md-2 col-lg-2 col-xl-2 mx-auto mb-4">
          <h6 class="text-uppercase fw-bold mb-4" style="color: #00b09b;">Quick Links</h6>
          <p><a href="/home" class="text-light text-decoration-none" style="opacity: 0.85;">Home</a></p>
          <p><a href="/mealplan" class="text-light text-decoration-none" style="opacity: 0.85;">Meal Plan</a></p>
          <p><a href="/recipes" class="text-light text-decoration-none" style="opacity: 0.85;">Recipes</a></p>
          <p><a href="/contact" class="text-light text-decoration-none" style="opacity: 0.85;">Contact</a></p>
        </div>
        <div class="col-md-4 col-lg-3 col-xl-3 mx-auto mb-md-0 mb-4">
          <h6 class="text-uppercase fw-bold mb-4" style="color: #00b09b;">Contact</h6>
          <p style="font-size: 0.95rem; opacity: 0.85;">New Delhi, India</p>
          <p style="font-size: 0.95rem; opacity: 0.85;">info@fitmeal.com</p>
          <p style="font-size: 0.95rem; opacity: 0.85;">+91 98765 43210</p>
        </div>
      </div>
    </div>
    <div class="text-center p-3 mt-4" style="background-color: rgba(0, 0, 0, 0.2); font-size: 0.9rem;">
      &copy; 2025 FitMeal. All Rights Reserved.
    </div>
  </footer>"""

def update_templates():
    for fname in os.listdir(TEMPLATE_DIR):
        if not fname.endswith('.html'):
            continue
        filepath = os.path.join(TEMPLATE_DIR, fname)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        li_pattern = re.compile(
            r'<li\s+class="nav-item ms-3">\s*<a[^>]*href="[/]"[^>]*>Sign Up</a>\s*</li>',
            re.IGNORECASE | re.DOTALL
        )
        content = li_pattern.sub(f'<li class="nav-item ms-3">\n{new_nav_button}\n        </li>', content)
        
        sidebar_a_pattern = re.compile(
            r'<li\s+class="nav-item ms-3">\s*<a[^>]*href="[/]"[^>]*class="btn btn-light text-success[^"]*"[^>]*>Sign Up</a>\s*</li>',
            re.IGNORECASE | re.DOTALL
        )
        content = sidebar_a_pattern.sub(f'<li class="nav-item ms-3">\n{new_nav_button}\n        </li>', content)
        
        # Another pattern for sidebar standalone links without <li> wrapping 
        sidebar_standalone_pattern = re.compile(
            r'<a[^>]*href="[/]"[^>]*class="btn btn-light text-success[^"]*"[^>]*>Sign Up</a>',
            re.IGNORECASE
        )
        content = sidebar_standalone_pattern.sub(new_nav_button, content)
        
        footer_pattern = re.compile(r'<footer.*?>.*?</footer>', re.IGNORECASE | re.DOTALL)
        if '<footer' in content.lower():
            content = footer_pattern.sub(new_footer, content)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    print("HTML files updated successfully")

if __name__ == '__main__':
    update_templates()
