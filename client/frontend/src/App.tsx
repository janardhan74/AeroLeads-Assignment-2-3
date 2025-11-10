import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import Navbar from './components/Navbar';
import BlogsList from './pages/BlogsList';
import BlogDetail from './pages/BlogDetail';
import GenerateBlog from './pages/GenerateBlog';
import VoiceAgent from './pages/VoiceAgent';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Navigate to="/blogs" replace />} />
          <Route path="/blogs" element={<BlogsList />} />
          <Route path="/blogs/:slug" element={<BlogDetail />} />
          <Route path="/generate_blog" element={<GenerateBlog />} />
          <Route path="/voice_agent" element={<VoiceAgent />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App
