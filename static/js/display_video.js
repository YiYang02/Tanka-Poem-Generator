// display_video.js

// Displays the tanka poem and read button 1 second after generation
function displayContent() {
    document.getElementById('tanka-poem').style.display = 'block';
    document.getElementById('read-button').style.display = 'inline'; // Change to 'block' if you prefer
}

 // Wait for 1 seconds (1000 milliseconds) before displaying the poem and the read button
 window.onload = function() {
    setTimeout(displayContent, 1000);
};

// Goes back to index.html frm display_video.html
function goBack() {
    window.location.href = '/';
}

// Text to Speech Function
function readPoem() {
    var poem = document.getElementById('tanka-poem').innerText;
    var msg = new SpeechSynthesisUtterance(poem);
    // Set properties for the speech synthesis
    msg.pitch = 1.0; // Pitch can be between 0 and 2
    msg.rate = 0.8; // Rate can be between 0.1 and 10
    // Select a voice; this is browser and system dependent
    msg.voice = speechSynthesis.getVoices().filter(function(voice) { return voice.name.includes('Female Asian American English'); })[0];
    
    speechSynthesis.speak(msg);
}