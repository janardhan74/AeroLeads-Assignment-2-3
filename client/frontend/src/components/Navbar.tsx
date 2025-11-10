import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
  const location = useLocation();

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          AeroLeads - Janardhan
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            color="inherit"
            component={Link}
            to="/blogs"
            sx={{
              backgroundColor: location.pathname === '/blogs' ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
            }}
          >
            Blogs
          </Button>
          <Button
            color="inherit"
            component={Link}
            to="/generate_blog"
            sx={{
              backgroundColor: location.pathname === '/generate_blog' ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
            }}
          >
            Generate Blog
          </Button>
          <Button
            color="inherit"
            component={Link}
            to="/voice_agent"
            sx={{
              backgroundColor: location.pathname === '/voice_agent' ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
            }}
          >
            Voice Agent
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;



