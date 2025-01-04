// External JavaScript file
function submitForm() {
    const companyName = document.getElementById('company').value;
    const fileInput = document.getElementById('companyData');
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('resultText');

    if (companyName && fileInput.files.length > 0) {
        resultText.textContent = `Data for ${companyName} has been uploaded successfully.`;
        resultDiv.style.display = 'block';

        // Mocking an upload and redirect
        setTimeout(() => {
            // Redirect to inventory predictor page
            window.location.href = "inventory_predictor.html"; // Change to your inventory predictor URL
        }, 2000); // Redirect after 2 seconds

        const formData = new FormData();
        formData.append('company', companyName);
        formData.append('companyData', fileInput.files[0]);

        /*
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
        })
        .catch(error => console.error('Error:', error));
        */
    } else {
        alert('Please enter a company name and select a file.');
    }
}
