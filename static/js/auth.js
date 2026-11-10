let isLoginMode = true;
let currentUser = null;
let isDemoMode = true;

function getAuthHeaders() {
  const token = localStorage.getItem("ragify_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function openAuthModal() {
  const errorEl = document.getElementById("authError");
  if (errorEl) errorEl.textContent = "";

  const modal = document.getElementById("authModal");
  if (modal) modal.classList.add("active");

  const emailInput = document.getElementById("authEmail");
  if (emailInput) emailInput.focus();
}

function closeAuthModal() {
  const modal = document.getElementById("authModal");
  if (modal) modal.classList.remove("active");
}

function handleOverlayClick(e) {
  if (e && e.target && e.target.id === "authModal") {
    closeAuthModal();
  }
}

function toggleProfileDropdown(e) {
  if (e) e.stopPropagation();
  const dropdown = document.getElementById("profileDropdown");
  if (dropdown) dropdown.classList.toggle("open");
}

window.addEventListener("click", () => {
  const dropdown = document.getElementById("profileDropdown");
  if (dropdown) dropdown.classList.remove("open");
});

function toggleAuthMode() {
  isLoginMode = !isLoginMode;

  const modalTitle = document.getElementById("authModalTitle");
  const submitBtn = document.getElementById("authSubmitBtn");
  const toggleText = document.getElementById("authToggleText");
  const nameFields = document.getElementById("signupNameFields");
  const confirmPassField = document.getElementById("confirmPasswordField");
  const errorEl = document.getElementById("authError");

  if (errorEl) errorEl.textContent = "";

  if (isLoginMode) {
    if (modalTitle) modalTitle.textContent = "Sign In";
    if (submitBtn) submitBtn.textContent = "Sign In";
    if (toggleText) toggleText.textContent = "Don't have an account? Sign Up";
    if (nameFields) nameFields.style.display = "none";
    if (confirmPassField) confirmPassField.style.display = "none";
  } else {
    if (modalTitle) modalTitle.textContent = "Create Account";
    if (submitBtn) submitBtn.textContent = "Sign Up";
    if (toggleText) toggleText.textContent = "Already have an account? Sign In";
    if (nameFields) nameFields.style.display = "flex";
    if (confirmPassField) confirmPassField.style.display = "flex";
  }
}

function handleAuthKey(e) {
  if (e.key === "Enter") {
    e.preventDefault();
    submitAuth(e);
  }
}

async function submitAuth(e) {
  if (e && e.preventDefault) e.preventDefault();

  const emailEl = document.getElementById("authEmail");
  const passEl = document.getElementById("authPassword");
  const errorEl = document.getElementById("authError");
  const btn = document.getElementById("authSubmitBtn");

  const email = emailEl ? emailEl.value.trim() : "";
  const password = passEl ? passEl.value : "";

  if (errorEl) errorEl.textContent = "";

  if (!email || !password) {
    if (errorEl) errorEl.textContent = "Please fill in all required fields.";
    return;
  }

  let payload = { email, password };

  if (!isLoginMode) {
    const firstNameEl = document.getElementById("authFirstName");
    const lastNameEl = document.getElementById("authLastName");
    const confirmPassEl = document.getElementById("authConfirmPassword");

    const firstName = firstNameEl ? firstNameEl.value.trim() : "";
    const lastName = lastNameEl ? lastNameEl.value.trim() : "";
    const confirmPassword = confirmPassEl ? confirmPassEl.value : "";

    if (!firstName) {
      if (errorEl) errorEl.textContent = "Please enter your first name.";
      return;
    }

    if (password !== confirmPassword) {
      if (errorEl) errorEl.textContent = "Passwords do not match.";
      return;
    }

    payload.first_name = firstName;
    if (lastName) payload.last_name = lastName;
  }

  btn.disabled = true;
  btn.textContent = "Processing...";

  const endpoint = isLoginMode ? "/auth/login" : "/auth/signup";

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      let message = "Authentication failed.";
      if (typeof data.detail === "string") {
        message = data.detail;
      } else if (Array.isArray(data.detail) && data.detail.length > 0) {
        message = data.detail[0].msg || JSON.stringify(data.detail[0]);
      }
      throw new Error(message);
    }

    if (data.access_token) {
      localStorage.setItem("ragify_token", data.access_token);
    }

    setLoggedInState(data.user);
    closeAuthModal();

    if (passEl) passEl.value = "";
    const confirmPassEl = document.getElementById("authConfirmPassword");
    if (confirmPassEl) confirmPassEl.value = "";
  } catch (err) {
    if (errorEl) errorEl.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = isLoginMode ? "Sign In" : "Sign Up";
  }
}

async function handleLogout(e) {
  if (e && e.stopPropagation) e.stopPropagation();

  try {
    await fetch("/auth/logout", {
      method: "POST",
      headers: getAuthHeaders(),
      credentials: "include",
    });
  } catch (err) {
    console.warn("Backend logout request failed:", err);
  }

  localStorage.removeItem("ragify_token");
  setDemoModeState();
}

async function checkSession() {
  try {
    const res = await fetch("/auth/me", {
      headers: getAuthHeaders(),
      credentials: "include",
    });

    if (!res.ok) {
      setDemoModeState();
      return;
    }

    const data = await res.json();
    if (data.authenticated && data.user) {
      setLoggedInState(data.user);
    } else {
      setDemoModeState();
    }
  } catch {
    setDemoModeState();
  }
}

function setLoggedInState(user) {
  if (!user) return;
  currentUser = user;
  isDemoMode = false;

  const authBtn = document.getElementById("authBtn");
  const demoBadge = document.getElementById("demoBadge");
  const profileContainer = document.getElementById("profileContainer");

  if (demoBadge) demoBadge.style.display = "none";
  if (authBtn) authBtn.style.display = "none";
  if (profileContainer) profileContainer.style.display = "block";

  let displayName = user.first_name
    ? user.last_name
      ? `${user.first_name} ${user.last_name}`
      : user.first_name
    : user.email
      ? user.email.split("@")[0]
      : "User";

  const initial = (user.first_name || displayName || "U")
    .trim()
    .charAt(0)
    .toUpperCase();
  const roleName = user.role === 2 || user.role === "admin" ? "ADMIN" : "USER";

  const userAvatar = document.getElementById("userAvatar");
  const headerUserName = document.getElementById("headerUserName");
  const dropdownEmail = document.getElementById("dropdownEmail");
  const dropdownRoleBadge = document.getElementById("dropdownRoleBadge");

  if (userAvatar) userAvatar.textContent = initial;
  if (headerUserName) headerUserName.textContent = displayName;
  if (dropdownEmail) dropdownEmail.textContent = user.email || "";
  if (dropdownRoleBadge) {
    dropdownRoleBadge.textContent = roleName;
    dropdownRoleBadge.style.display = "inline-block";
  }
}

function setDemoModeState() {
  currentUser = null;
  isDemoMode = true;

  const authBtn = document.getElementById("authBtn");
  const demoBadge = document.getElementById("demoBadge");
  const profileContainer = document.getElementById("profileContainer");
  const userAvatar = document.getElementById("userAvatar");
  const headerUserName = document.getElementById("headerUserName");
  const dropdownEmail = document.getElementById("dropdownEmail");
  const dropdownRoleBadge = document.getElementById("dropdownRoleBadge");

  if (userAvatar) userAvatar.textContent = "";
  if (headerUserName) headerUserName.textContent = "";
  if (dropdownEmail) dropdownEmail.textContent = "";
  if (dropdownRoleBadge) {
    dropdownRoleBadge.textContent = "";
    dropdownRoleBadge.style.display = "none";
  }

  if (demoBadge) demoBadge.style.display = "inline-flex";
  if (authBtn) authBtn.style.display = "inline-flex";
  if (profileContainer) profileContainer.style.display = "none";

  const dropdown = document.getElementById("profileDropdown");
  if (dropdown) dropdown.classList.remove("open");
}

document.addEventListener("DOMContentLoaded", () => {
  checkSession();
});
