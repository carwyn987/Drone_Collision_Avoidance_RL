var canvas = document.querySelector("canvas");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

var l = canvas.getContext('2d');

// x and y are the initial coordinates of the circle
// vx and vy are the initial respective speeds
var x = 200
var y = 200;
var vx = 5;
var vy = 0;
var radius = 20;

move();

// This function will do the animation
function move() {
    requestAnimationFrame(move);

    // It clears the specified pixels within
    // the given rectangle
    //l.clearRect(0, 0, innerWidth, innerHeight);
    l.clearRect(x - radius - vx - 1, y - radius - vy - 1, 2*radius + 2, 2*radius + 2);

    // Creating a circle
    l.beginPath();
    l.strokeStyle = "black";
    l.arc(x, y, radius, 0, Math.PI * 2, false);
    l.stroke();

    // Conditions sso that the ball bounces
    // from the edges
    if (radius + x > innerWidth)
        vx = 0 - vx;

    if (x - radius < 0)
        vx = 0 - vx;

    if (y + radius > innerHeight)
        vy = 0 - vy - 1;

    if (y - radius < 0)
        vy = 0 - vy;

    vy = vy + 1;
    x = x + vx;
    y = y + vy;
}