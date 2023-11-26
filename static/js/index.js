
// Ensures that only can click upload button if user uploads a video file
function checkFile() {
    if (document.getElementById('video-file').value) {
        document.getElementById('upload-btn').disabled = false;
    } else {
        document.getElementById('upload-btn').disabled = true;
    }
}

// Redirects to the view_poem route with the selected poem ID
function goToPoem() {
    var selectedPoem = document.getElementById('poem-selector').value;
    window.location.href = `/view_poem?poem_id=${selectedPoem}`;
}

// This function should be called after the server has processed the video
// and returned the list of detected objects
function processVideoResult(objectsDetected) {
    promptDescription(objectsDetected);
}
