const puppeteer = require('puppeteer');
const { spawn } = require('child_process');

const [,, message, thickness, fontSize, startX, startY, fontFace, textColor, maxTextWidthPct, canvasWidth, canvasHeight, rotateX, rotateY, rotateZ, running_port] = process.argv;

(async () => {
    // Start HTTP server
    console.time('Starting HTTP Server');
    const server = spawn('python3', ['-m', 'http.server', '8170'], {
        cwd: process.cwd(), // Set the current working directory
        detached: true, // Detach the process so it can run independently
        stdio: 'ignore' // Ignore stdio
    });
    // server.unref(); // Don't wait for the spawned process to exit
    console.log('HTTP server started on port 8170');
    console.timeEnd('Starting HTTP Server');

    // Give the server a moment to start up
    await new Promise(resolve => setTimeout(resolve, 1000));

    console.time('Total Execution Time');

    console.time('Launching Browser');
    const browser = await puppeteer.launch({
        headless: true // Ensure it runs in headless mode
    });
    console.timeEnd('Launching Browser');

    const page = await browser.newPage();

    console.time('Loading Page');
    let params = {
        message: encodeURIComponent(message),
        thickness: parseInt(thickness),
        fontSize: parseInt(fontSize),
        startX: parseFloat(startX),
        startY: parseFloat(startY),
        fontFace: encodeURIComponent(fontFace),
        textColor: encodeURIComponent(textColor),
        maxTextWidthPct: parseFloat(maxTextWidthPct),
        canvasWidth: parseInt(canvasWidth),
        canvasHeight: parseInt(canvasHeight),
        rotateX: parseFloat(rotateX),
        rotateY: parseFloat(rotateY),
        rotateZ: parseFloat(rotateZ),
    };

    // Construct the query string from the parameters object
    let queryString = Object.keys(params)
                            .map(key => `${key}=${params[key]}`)
                            .join('&');

    let url = `http://localhost:8170/sketch.html?${queryString}`;
    await page.goto(url, {waitUntil: 'networkidle0'});

    console.timeEnd('Loading Page');

    // Dynamic text to be displayed in the sketch
    const dynamicText = "Hello, Dynamic World!";

    console.time('Injecting Text and Redrawing');
    // Inject dynamic text into the sketch
    await page.evaluate((text) => {
        window.dynamicText = text;
        redraw(); // Assuming redraw() is properly defined in your sketch
    }, dynamicText);
    await page.waitForFunction(() => window.sketchReady === true);

    console.timeEnd('Injecting Text and Redrawing');

    console.time('Taking Screenshot');
    await page.setViewport({
        width: params.canvasWidth,
        height: params.canvasHeight
    });

    // await page.screenshot({path: 'p5-sketch-dynamic.png'});
    await page.screenshot({
        path: 'p5-sketch-dynamic.png',
        omitBackground: true // This ensures the background is considered transparent
    });
    
    
    console.timeEnd('Taking Screenshot');

    await browser.close();
    console.log('p5.js sketch with dynamic text rendered and saved as p5-sketch-dynamic.png');

    console.timeEnd('Total Execution Time');

    // Optionally, stop the server if you no longer need it
    server.kill();
})();
