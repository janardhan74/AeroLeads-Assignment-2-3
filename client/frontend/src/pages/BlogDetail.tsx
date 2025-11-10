import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Button,
  Paper,
} from '@mui/material';
import ReactMarkdown from 'react-markdown';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface BlogData {
  slug: string;
  metadata: {
    title?: string;
    summary?: string;
    created_at?: string;
    reading_time_min?: number;
    [key: string]: any;
  };
  content: string;
}

const BlogDetail = () => {
  const { slug } = useParams<{ slug: string }>();
  const navigate = useNavigate();
  const [blog, setBlog] = useState<BlogData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (slug) {
      fetchBlog(slug);
    }
  }, [slug]);

  const fetchBlog = async (blogSlug: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE_URL}/api/blog/${blogSlug}`);
      if (!response.ok) {
        throw new Error('Failed to fetch blog');
      }
      const data = await response.json();
      setBlog(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
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
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/blogs')}
          sx={{ mt: 2 }}
        >
          Back to Blogs
        </Button>
      </Container>
    );
  }

  if (!blog) {
    return (
      <Container sx={{ mt: 4 }}>
        <Alert severity="warning">Blog not found</Alert>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/blogs')}
          sx={{ mt: 2 }}
        >
          Back to Blogs
        </Button>
      </Container>
    );
  }

  return (
    <Container sx={{ mt: 4, mb: 4, maxWidth: '900px' }}>
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate('/blogs')}
        sx={{ mb: 3 }}
      >
        Back to Blogs
      </Button>
      
      <Paper elevation={2} sx={{ p: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          {blog.metadata.title || blog.slug}
        </Typography>
        
        {blog.metadata.summary && (
          <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
            {blog.metadata.summary}
          </Typography>
        )}
        
        <Box sx={{ display: 'flex', gap: 3, mb: 4, pb: 2, borderBottom: 1, borderColor: 'divider' }}>
          {blog.metadata.reading_time_min && (
            <Typography variant="body2" color="text.secondary">
              {blog.metadata.reading_time_min} min read
            </Typography>
          )}
          {blog.metadata.created_at && (
            <Typography variant="body2" color="text.secondary">
              {new Date(blog.metadata.created_at).toLocaleDateString()}
            </Typography>
          )}
        </Box>
        
        <Box
          sx={{
            '& h1, & h2, & h3, & h4, & h5, & h6': {
              mt: 3,
              mb: 2,
            },
            '& p': {
              mb: 2,
              lineHeight: 1.8,
            },
            '& ul, & ol': {
              mb: 2,
              pl: 3,
            },
            '& li': {
              mb: 1,
            },
            '& code': {
              backgroundColor: 'rgba(0, 0, 0, 0.05)',
              padding: '2px 6px',
              borderRadius: '4px',
              fontFamily: 'monospace',
            },
            '& pre': {
              backgroundColor: 'rgba(0, 0, 0, 0.05)',
              padding: 2,
              borderRadius: '4px',
              overflow: 'auto',
              mb: 2,
            },
            '& blockquote': {
              borderLeft: '4px solid',
              borderColor: 'primary.main',
              pl: 2,
              ml: 0,
              fontStyle: 'italic',
              mb: 2,
            },
          }}
        >
          <ReactMarkdown>{blog.content}</ReactMarkdown>
        </Box>
      </Paper>
    </Container>
  );
};

export default BlogDetail;



