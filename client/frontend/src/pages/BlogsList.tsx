import { useEffect, useState } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Box,
  CircularProgress,
  Alert,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';

interface BlogMetadata {
  slug: string;
  title?: string;
  summary?: string;
  created_at?: string;
  reading_time_min?: number;
  filename?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const BlogsList = () => {
  const [blogs, setBlogs] = useState<BlogMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchBlogs();
  }, []);

  const fetchBlogs = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE_URL}/api/blogs`);
      if (!response.ok) {
        throw new Error('Failed to fetch blogs');
      }
      const data = await response.json();
      setBlogs(data.blogs || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleBlogClick = (slug: string) => {
    navigate(`/blogs/${slug}`);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container sx={{ mt: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  return (
    <Container sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        All Blogs
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        {blogs.length} blog{blogs.length !== 1 ? 's' : ''} available
      </Typography>
      <Grid container spacing={3}>
        {blogs.length === 0 ? (
          <Grid item xs={12}>
            <Alert severity="info">No blogs available. Generate your first blog!</Alert>
          </Grid>
        ) : (
          blogs.map((blog) => (
            <Grid item xs={12} sm={6} md={4} key={blog.slug}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4,
                  },
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h5" component="h2" gutterBottom>
                    {blog.title || blog.slug}
                  </Typography>
                  {blog.summary && (
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {blog.summary.length > 150
                        ? `${blog.summary.substring(0, 150)}...`
                        : blog.summary}
                    </Typography>
                  )}
                  <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
                    {blog.reading_time_min && (
                      <Typography variant="caption" color="text.secondary">
                        {blog.reading_time_min} min read
                      </Typography>
                    )}
                    {blog.created_at && (
                      <Typography variant="caption" color="text.secondary">
                        {new Date(blog.created_at).toLocaleDateString()}
                      </Typography>
                    )}
                  </Box>
                </CardContent>
                <CardActions>
                  <Button
                    size="small"
                    variant="contained"
                    onClick={() => handleBlogClick(blog.slug)}
                  >
                    Read More
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))
        )}
      </Grid>
    </Container>
  );
};

export default BlogsList;



