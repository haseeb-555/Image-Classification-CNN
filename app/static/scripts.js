// script.js

document
  .getElementById("uploadForm")
  .addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent form submission

    const fileInput = document.getElementById("fileInput");
    const resultDiv = document.getElementById("result");
    const predictionText = document.getElementById("predictionText");

    // Show loading message
    resultDiv.classList.remove("hidden");
    predictionText.textContent = "Loading...";

    // Get the uploaded file
    const file = fileInput.files[0];
    if (!file) {
      alert("Please select a file.");
      return;
    }

    // Create a FormData object
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Send the file to the Flask API
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Error in prediction API");
      }

      const data = await response.json();

      // Display the prediction
      if (data.prediction) {
        predictionText.textContent = `This is likely a ${data.prediction}.`;
      } else {
        predictionText.textContent = "Error: Could not get prediction.";
      }
    } catch (error) {
      predictionText.textContent = `Error: ${error.message}`;
    }
  });
