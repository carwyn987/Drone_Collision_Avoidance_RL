droneConstructor = function(x,y,vx,vy){
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.angle = 0;

    this.rotateSpeed = 2;
    this.accelerateSpeed = 5;
}

ballConstructor = function(x,y,vx,vy,radius){
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.radius = radius;
}

// DEFINE GAME VARIABLES
let marginOfIntersection = 10;
let gravity = .2;
let drone = new droneConstructor(500,300,0,0);
let ball = new ballConstructor(0,400,10,-10,30);
let score = 0;

window.onload = function(){
    var canvas = document.querySelector("canvas");

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    var l = canvas.getContext('2d');

    window.addEventListener("keydown", keypressed_handler, false);
    let scoreVal = document.getElementById("scoreVal");

    var moveInterval = setInterval(function(){
        draw(l);
    }, 25);
}

draw = function(l){
    l.clearRect(0, 0, innerWidth, innerHeight);
    var droneImg = new Image();
    droneImg.src="./images/drone4.png";

    l.save();
    l.translate( drone.x+droneImg.width/2, drone.y+droneImg.height/2 );
    l.rotate( drone.angle );
    l.translate( -drone.x-droneImg.width/2, -drone.y-droneImg.height/2 );
    l.drawImage(droneImg, drone.x, drone.y);
    l.restore();

    l.beginPath();
    l.arc(ball.x, ball.y, ball.radius, 0, 2 * Math.PI);
    l.fillStyle = "red";
    l.fill();
    l.stroke();
    
    drone.x += drone.vx;
    drone.y += drone.vy;
    drone.vy += gravity;

    ball.x += ball.vx;
    ball.y += ball.vy;
    ball.vy += gravity;

    // Lower boundary of canvas: drone
    if(droneImg.height + drone.y >= innerHeight){
        drone.vy = -1;
    }
    // Upper boundary of canvas: drone
    if(drone.y <= 0){
        drone.vy = 1;
    }
    // Right boundary of canvas: drone
    if(droneImg.width + drone.x >= innerWidth){
        drone.vx = 0;
    }
    // Left boundary of canvas: drone
    if(drone.x <= 0){
        drone.vx = 0;
    }
    // boundary of canvas: ball
    if(ball.y > 200+innerHeight || ball.y < -30 || ball.x > 100+innerWidth || ball.x < -30){
        ball.x = 0;
        ball.y = Math.random()*innerHeight;
        ball.vx = Math.random()*20 + 10;
        ball.vy = Math.random()*10 - 5;

        score += 2;
        scoreVal.innerHTML = score;
    }

    // intersection of ball and drone:
    if(ball.x + 2 * ball.radius - marginOfIntersection > drone.x && ball.x + marginOfIntersection < drone.x + droneImg.width){
        if(ball.y + 2 * ball.radius > drone.y + marginOfIntersection && ball.y < drone.y + droneImg.height - marginOfIntersection){
            score -= 1;
            scoreVal.innerHTML = score;
        }
    }
}

keypressed_handler = function(event){
    //left
    if(event.keyCode == 37){
        drone.angle -= .2;
        if(drone.angle>2*Math.PI){
            drone.angle -= 2*Math.PI;
        }
        if(drone.angle<0){
            drone.angle += 2*Math.PI;
        }
    }
    //up
    if(event.keyCode == 38){
        drone.vy -= 5*Math.cos(drone.angle);
        drone.vx += 5*Math.sin(drone.angle);
    }
    //right
    if(event.keyCode == 39){
        drone.angle += .2;
        if(drone.angle>2*Math.PI){
            drone.angle -= 2*Math.PI;
        }
        if(drone.angle<0){
            drone.angle += 2*Math.PI;
        }
    }
    //down
    if(event.keyCode == 40){
        drone.vy += 5*Math.cos(drone.angle);
        drone.vx -= 5*Math.sin(drone.angle);
    }
}