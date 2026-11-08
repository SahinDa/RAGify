const dropzone = document.getElementById("dropzone");
const dropzoneIcon = document.getElementById("dropzoneIcon");
const dropzoneText = document.getElementById("dropzoneText");

function handleDropzoneClick() {
  if (!currentUser) {
    openAuthModal();
    return;
  }
  document.getElementById("fileInput").click();
}

if (dropzone) {
  ["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((name) => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (!currentUser) {
      openAuthModal();
      return;
    }
    if (e.dataTransfer.files.length) {
      handleSingleFile(e.dataTransfer.files[0]);
    }
  });
}

async function handleSingleFile(file) {
  if (!file) return;

  if (!currentUser) {
    openAuthModal();
    return;
  }

  dropzone.classList.remove("success");
  dropzoneIcon.textContent = "⚡";
  dropzoneText.innerHTML = `Indexing <strong>${file.name}</strong>...`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload", {
      method: "POST",
      headers: getAuthHeaders(),
      body: formData,
    });
    const data = await res.json();

    if (res.ok) {
      dropzone.classList.add("success");
      dropzoneIcon.textContent = "✅";
      dropzoneText.innerHTML = `<strong>Upload done!</strong>`;

      setTimeout(() => {
        dropzone.classList.remove("success");
        dropzoneIcon.textContent = "📄";
        dropzoneText.innerHTML = `Drop document here or <strong>browse</strong> (.pdf, .docx, .txt)`;
      }, 3000);
    } else {
      alert(`Failed to index ${file.name}: ` + (data.detail || "Server error"));
      dropzoneIcon.textContent = "📄";
      dropzoneText.innerHTML = `Drop document here or <strong>browse</strong> (.pdf, .docx, .txt)`;
    }
  } catch (err) {
    alert(`Error uploading ${file.name}: ` + err.message);
    dropzoneIcon.textContent = "📄";
    dropzoneText.innerHTML = `Drop document here or <strong>browse</strong> (.pdf, .docx, .txt)`;
  } finally {
    document.getElementById("fileInput").value = "";
  }
}
