import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { exec } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Custom plugin to handle model training requests
function trainingServerPlugin() {
  return {
    name: 'training-server',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url && req.url.startsWith('/api/train')) {
          const urlParams = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
          const symbol = urlParams.searchParams.get('symbol') || 'AAPL';
          const horizon = urlParams.searchParams.get('horizon') || '5';
          const interval = urlParams.searchParams.get('interval') || '1d';
          const trials = urlParams.searchParams.get('trials') || '30';
          const noWalkforward = urlParams.searchParams.get('no_walkforward') || 'true';
          
          res.writeHead(200, { 
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache' 
          });
          
          // Resolve python path inside virtual environment
          let pythonCmd = 'python';
          const venvWin = path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe');
          const venvUnix = path.join(__dirname, '..', 'venv', 'bin', 'python');
          if (fs.existsSync(venvWin)) {
            pythonCmd = `.\\venv\\Scripts\\python.exe`;
          } else if (fs.existsSync(venvUnix)) {
            pythonCmd = `./venv/bin/python`;
          }
          
          // Build the exact python run command
          const wfFlag = noWalkforward === 'true' ? '--no-walkforward' : '';
          const command = `${pythonCmd} run_mvp.py --symbol ${symbol} --horizon ${horizon} --interval ${interval} --tuning-trials ${trials} ${wfFlag}`;
          
          console.log(`[Vite Backend] Running training command: ${command}`);
          
          // Execute python script from parent folder Cwd
          exec(command, { cwd: '../' }, (error, stdout, stderr) => {
            if (error) {
              console.error(`[Vite Backend] Command failed: ${error.message}`);
              console.error(stderr);
              res.end(JSON.stringify({ 
                success: false, 
                error: error.message,
                stderr: stderr 
              }));
              return;
            }
            console.log(`[Vite Backend] Training successfully completed for ${symbol}!`);
            res.end(JSON.stringify({ success: true }));
          });
          
          return;
        }
        next();
      });
    }
  };
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), trainingServerPlugin()],
})
