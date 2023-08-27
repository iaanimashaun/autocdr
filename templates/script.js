// document.addEventListener("DOMContentLoaded", function () {
//     const imageForm = document.getElementById("imageForm");
//     const message = document.getElementById("message");
//     const result = document.getElementById("result");

//     imageForm.addEventListener("submit", async (e) => {
//         e.preventDefault();

//         const formData = new FormData(imageForm);

//         try {
//             const response = await fetch("/upload", {
//                 method: "POST",
//                 body: formData,
//             });
    
//             const data = await response.json();
    
//             message.innerHTML = data.message;
    
//             if (data.cdr !== undefined) {
//                 result.innerHTML = `Calculated Cup-Disc Ratio: ${data.cdr}`;
//                 // Redirect to result page
//                 window.location.href = `/result/${data.cdr}`;
//             } else {
//                 result.innerHTML = '';  // Clear the result if CDR is not provided
//             }
    
//             imageForm.reset();
//         } catch (error) {
//             console.error("Error uploading image:", error);
//             message.innerHTML = "Error uploading image.";
//         }
//     });
// });








document.addEventListener("DOMContentLoaded", function () {
    const imageForm = document.getElementById("imageForm");
    const message = document.getElementById("message");
    const result = document.getElementById("result");
    const defaultImages = document.querySelectorAll(".default-image");
    const selectedDefaultImageInput = document.getElementById("selectedDefaultImage");

    defaultImages.forEach(image => {
        image.addEventListener("click", function () {
            const imageSrc = image.getAttribute("data-src");
            selectedDefaultImageInput.value = imageSrc;
        });
    });

    imageForm.addEventListener("submit", async function (e) {
        e.preventDefault();

        const formData = new FormData(imageForm);

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            message.innerHTML = data.message;

            if (data.cdr !== undefined) {
                result.innerHTML = `Calculated Cup-Disc Ratio: ${data.cdr}`;
                // Redirect to result page
                window.location.href = `/result/${data.cdr}`;
            } else {
                result.innerHTML = '';  // Clear the result if CDR is not provided
            }

            imageForm.reset();
            selectedDefaultImageInput.value = ''; // Reset selected default image
        } catch (error) {
            console.error("Error uploading image:", error);
            message.innerHTML = "Error uploading image.";
        }
    });
});







// Function to select a default image
function selectDefaultImage(element) {
    var defaultImageSrc = element.getAttribute("data-src");
    document.getElementById("selectedDefaultImage").value = defaultImageSrc;
    
    // Highlight the selected default image visually (optional)
    var defaultImages = document.querySelectorAll(".default-image");
    defaultImages.forEach(img => img.classList.remove("selected"));
    element.classList.add("selected");
}

// Attach event listener to the "Upload" button
document.getElementById("imageForm").addEventListener("submit", function (event) {
    var selectedDefaultImage = document.getElementById("selectedDefaultImage").value;
    if (!selectedDefaultImage && !document.getElementById("image").files[0]) {
        event.preventDefault(); // Prevent form submission if no image is selected
        document.getElementById("message").textContent = "Please select an image or choose a default image.";
    }
});



