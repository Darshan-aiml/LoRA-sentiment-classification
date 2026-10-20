/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#090d16',
        surface: {
          DEFAULT: '#0f172a',
          hover: '#1e293b',
          border: '#1e293b',
          muted: '#334155',
        },
        brand: {
          DEFAULT: '#3b82f6',
          hover: '#2563eb',
          subtle: '#1d4ed8',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      }
    },
  },
  plugins: [],
}
