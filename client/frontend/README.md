# Frontend - AeroLeads App

Modern React-based frontend application for AI Blog Generation and Voice Agent management.

## 🚀 Tech Stack

- **React 19** - UI library
- **TypeScript** - Type safety
- **Material-UI (MUI) v7** - Component library and design system
- **Vite** - Fast build tool and dev server
- **React Router v7** - Client-side routing
- **React Markdown** - Markdown rendering

## 📋 Prerequisites

- Node.js 18+ and npm

## 🛠️ Installation

```bash
# Install dependencies
npm install
```

## 🏃 Development

```bash
# Start development server
npm run dev
```

The application will be available at `http://localhost:5173`

## 📦 Build

```bash
# Build for production
npm run build
```

The production build will be in the `dist/` directory.

## 🧪 Linting

```bash
# Run ESLint
npm run lint
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── Navbar.tsx          # Navigation component
│   ├── pages/
│   │   ├── BlogsList.tsx        # Blog listing page
│   │   ├── BlogDetail.tsx       # Blog detail/view page
│   │   ├── GenerateBlog.tsx     # Blog generation form
│   │   └── VoiceAgent.tsx       # Voice agent interface
│   ├── App.tsx                  # Main app component with routing
│   ├── main.tsx                 # Entry point
│   └── index.css                # Global styles
├── public/                       # Static assets
├── package.json
├── vite.config.ts               # Vite configuration
└── tsconfig.json                # TypeScript configuration
```

## 🎨 Features

### Blog Management

- **Blog List**: View all generated blogs
- **Blog Detail**: Read full blog posts with markdown rendering
- **Blog Generation**: Create new blogs with customizable parameters

### Voice Agent

- **Make Call Tab**: Direct phone call interface

  - Phone number input
  - Optional room name and caller ID
  - Transcript timeout configuration
  - Real-time call status and transcript display

- **Agent Call Tab**: Natural language interface

  - Text prompt input (e.g., "make call to +1234567890")
  - Support for multiple numbers in one prompt
  - Agent response display
  - Call results with transcripts

- **Multiple Calls Tab**: Batch calling
  - File upload support (.txt, .csv)
  - Manual phone number entry
  - Batch results with individual call details
  - Success/failure statistics

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `frontend` directory:

```env
VITE_API_URL=http://localhost:8000
```

### API Integration

The frontend communicates with the backend API at the base URL specified in `VITE_API_URL` (defaults to `http://localhost:8000`).

**API Endpoints Used:**

- `POST /api/generate-articles` - Generate blogs
- `GET /api/blogs` - List blogs
- `GET /api/blogs/{slug}` - Get blog details
- `POST /api/voice/make-a-call` - Single call
- `POST /api/voice/make-agent-call` - Agent call
- `POST /api/voice/make-multiple-calls` - Batch calls

## 🎯 Key Components

### VoiceAgent.tsx

Main voice agent interface with three tabs:

- Tab-based navigation
- File upload for batch calls
- Real-time status updates
- Transcript display with expandable sections

### GenerateBlog.tsx

Blog generation form with:

- Topic input (required)
- Optional parameters (audience, tone, style, length)
- AI provider selection
- Content options (code, TLDR, further reading)

### BlogsList.tsx

Blog listing page with:

- Grid layout of blog cards
- Markdown preview
- Navigation to detail pages

### BlogDetail.tsx

Blog detail page with:

- Full markdown rendering
- Metadata display
- Navigation controls

## 🎨 Styling

- **Material-UI Theme**: Custom theme configuration in `App.tsx`
- **Responsive Design**: Mobile-first approach with MUI Grid system
- **Modern UI**: Gradient backgrounds, smooth animations, and clean layouts

## 🐛 Troubleshooting

### TypeScript Errors

If you see TypeScript errors related to MUI Grid components, they are suppressed with `@ts-expect-error` comments due to MUI v7 type definition issues. The components work correctly at runtime.

### CORS Issues

Ensure the backend CORS middleware allows your frontend origin (default: `http://localhost:5173`).

### API Connection

Verify that:

1. Backend server is running on the correct port
2. `VITE_API_URL` matches your backend URL
3. Backend CORS settings allow your frontend origin

## 📝 Development Notes

- Uses React 19 with latest features
- TypeScript strict mode enabled
- ESLint configured for React best practices
- Vite for fast HMR (Hot Module Replacement)

## 🚀 Production Deployment

1. Build the application:

   ```bash
   npm run build
   ```

2. Serve the `dist/` directory with a web server (nginx, Apache, etc.)

3. Update `VITE_API_URL` to point to your production API

4. Configure CORS on the backend to allow your production domain

## 📚 Additional Resources

- [React Documentation](https://react.dev/)
- [Material-UI Documentation](https://mui.com/)
- [Vite Documentation](https://vitejs.dev/)
- [TypeScript Documentation](https://www.typescriptlang.org/)
