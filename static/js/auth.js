let isLoginMode = true;
let currentUser = null;

function getAuthHeaders() {
  const token = localStorage.getItem("ragify_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function openAuthModal() {
  if (currentUser) {
    handleLogout();
    return;
  }
  document.getElementById("authError").textContent = "";
  document.getElementById("authModal").classList.add("active");
  document.getElementById("authEmail").focus();
}

function closeAuthModal() {
  document.getElementById("authModal").classList.remove("active");
}

function handleOverlayClick() {
  closeAuthModal();
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

  errorEl.textContent = "";

  if (isLoginMode) {
    modalTitle.textContent = "Sign In";
    submitBtn.textContent = "Sign In";
    toggleText.textContent = "Don't have an account? Sign Up";
    nameFields.style.display = "none";
    confirmPassField.style.display = "none";
  } else {
    modalTitle.textContent = "Create Account";
    submitBtn.textContent = "Sign Up";
    toggleText.textContent = "Already have an account? Sign In";
    nameFields.style.display = "flex";
    confirmPassField.style.display = "flex";
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

  const email = document.getElementById("authEmail").value.trim();
  const password = document.getElementById("authPassword").value;
  const errorEl = document.getElementById("authError");
  const btn = document.getElementById("authSubmitBtn");

  errorEl.textContent = "";

  if (!email || !password) {
    errorEl.textContent = "Please fill in all required fields.";
    return;
  }

  let payload = { email, password };

  if (!isLoginMode) {
    const firstName = document.getElementById("authFirstName").value.trim();
    const lastName = document.getElementById("authLastName").value.trim();
    const confirmPassword = document.getElementById(
      "authConfirmPassword",
    ).value;

    if (!firstName || !lastName) {
      errorEl.textContent = "Please enter both your first and last name.";
      return;
    }

    if (password !== confirmPassword) {
      errorEl.textContent = "Passwords do not match.";
      return;
    }

    payload.first_name = firstName;
    payload.last_name = lastName;
  }

  btn.disabled = true;
  btn.textContent = "Processing...";

  const endpoint = isLoginMode ? "/auth/login" : "/auth/signup";

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Authentication request failed.");
    }

    // Browser automatically sets the 24h cookie via HTTP Set-Cookie header.
    // Hydrate UI state directly from response payload:
    setLoggedInState(data.user);
    closeAuthModal();
  } catch (err) {
    errorEl.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = isLoginMode ? "Sign In" : "Sign Up";
  }
}

async function handleLogout() {
  const token = localStorage.getItem("ragify_token");

  // Call the backend logout route if a token exists
  if (token) {
    try {
      await fetch("/auth/logout", {
        method: "POST",
        headers: getAuthHeaders(),
      });
    } catch (err) {
      console.warn("Logout request failed:", err);
    }
  }

  // Clear local session and update UI
  localStorage.removeItem("ragify_token");
  setLoggedOutState();

  if (typeof resetConversation === "function") {
    resetConversation();
  }
}
async function handleLogout() {
  const token = localStorage.getItem("ragify_token");

  // Call the backend logout route if a token exists
  if (token) {
    try {
      await fetch("/auth/logout", {
        method: "POST",
        headers: getAuthHeaders(),
      });
    } catch (err) {
      console.warn("Logout request failed:", err);
    }
  }

  // Clear local session and update UI
  localStorage.removeItem("ragify_token");
  setLoggedOutState();

  if (typeof resetConversation === "function") {
    resetConversation();
  }
}

async function checkSession() {
  try {
    // Browser automatically passes the cookie along with this request
    const res = await fetch("/auth/me");
    if (!res.ok) throw new Error("No active session");
    const user = await res.json();
    setLoggedInState(user);
  } catch {
    setLoggedOutState();
  }
}

function setLoggedInState(user) {
  currentUser = user;

  // Show Profile section, hide Sign In button
  const authBtn = document.getElementById("authBtn");
  const profileContainer = document.getElementById("profileContainer");
  if (authBtn) authBtn.style.display = "none";
  if (profileContainer) profileContainer.style.display = "block";

  const displayName = user.email.split("@")[0];
  if (user.first_name) {
    displayName = user.last_name
      ? `${user.first_name} ${user.last_name}`
      : user.first_name;
  }
  const initial = displayName.charAt(0).toUpperCase();
  const roleName = user.role === 2 || user.role === "admin" ? "ADMIN" : "USER";

  const userAvatar = document.getElementById("userAvatar");
  const headerUserName = document.getElementById("headerUserName");
  const dropdownEmail = document.getElementById("dropdownEmail");
  const dropdownRoleBadge = document.getElementById("dropdownRoleBadge");

  if (userAvatar) userAvatar.textContent = initial;
  if (headerUserName) headerUserName.textContent = displayName;
  if (dropdownEmail) dropdownEmail.textContent = user.email;
  if (dropdownRoleBadge) dropdownRoleBadge.textContent = roleName;
}

function setLoggedOutState() {
  currentUser = null;

  const authBtn = document.getElementById("authBtn");
  const profileContainer = document.getElementById("profileContainer");
  if (authBtn) authBtn.style.display = "inline-flex";
  if (profileContainer) profileContainer.style.display = "none";

  const dropdown = document.getElementById("profileDropdown");
  if (dropdown) dropdown.classList.remove("open");
}

async function handleLogout() {
  if (e && e.stopPropagation) e.stopPropagation();

  try {
    // Calling backend logout tells the server to delete the cookie via response.delete_cookie()
    await fetch("/auth/logout", { method: "POST" });
  } catch (err) {
    console.warn("Backend logout request failed:", err);
  }

  setLoggedOutState();

  if (typeof resetConversation === "function") {
    resetConversation();
  }
}
