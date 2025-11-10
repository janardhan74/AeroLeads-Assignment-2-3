import { useState } from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  Alert,
  CircularProgress,
  Grid,
  FormHelperText,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface BlogRequest {
  topic: string;
  audience?: string;
  tone?: string;
  style?: string;
  length?: number;
  keywords?: string;
  include_code?: boolean;
  include_tldr?: boolean;
  include_further_reading?: boolean;
}

interface GeneratePayload {
  provider?: string;
  items: BlogRequest[];
}

const GenerateBlog = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  const [formData, setFormData] = useState<BlogRequest>({
    topic: '',
    audience: 'Developers',
    tone: 'Clear, friendly',
    style: 'How-to tutorial',
    length: 1400,
    keywords: '',
    include_code: true,
    include_tldr: true,
    include_further_reading: true,
  });
  
  const [provider, setProvider] = useState<string>('');
  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleChange = (field: keyof BlogRequest) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const value = e.target.value;
    if (field === 'length') {
      setFormData({ ...formData, [field]: value ? parseInt(value) : undefined });
    } else if (field === 'keywords' || field === 'topic' || field === 'audience' || field === 'tone' || field === 'style') {
      setFormData({ ...formData, [field]: value });
    }
    // Clear error for this field
    if (errors[field]) {
      setErrors({ ...errors, [field]: '' });
    }
  };

  const handleCheckboxChange = (field: keyof BlogRequest) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFormData({ ...formData, [field]: e.target.checked });
  };

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.topic.trim()) {
      newErrors.topic = 'Topic is required';
    }
    
    if (formData.length && formData.length < 100) {
      newErrors.length = 'Length must be at least 100 words';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!validateForm()) {
      return;
    }

    try {
      setLoading(true);
      
      // Prepare the payload
      const payload: GeneratePayload = {
        provider: provider || undefined,
        items: [
          {
            topic: formData.topic,
            audience: formData.audience || undefined,
            tone: formData.tone || undefined,
            style: formData.style || undefined,
            length: formData.length || undefined,
            keywords: formData.keywords || undefined,
            include_code: formData.include_code,
            include_tldr: formData.include_tldr,
            include_further_reading: formData.include_further_reading,
          },
        ],
      };

      const response = await fetch(`${API_BASE_URL}/api/generate-articles`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to generate blog' }));
        throw new Error(errorData.detail || 'Failed to generate blog');
      }

      const data = await response.json();
      
      setSuccess(`Successfully generated ${data.count} blog(s)!`);
      
      // Reset form
      setFormData({
        topic: '',
        audience: 'Developers',
        tone: 'Clear, friendly',
        style: 'How-to tutorial',
        length: 1400,
        keywords: '',
        include_code: true,
        include_tldr: true,
        include_further_reading: true,
      });
      setProvider('');

      // Redirect to the newly created blog if available, otherwise to blogs list
      if (data.created && data.created.length > 0 && data.created[0].meta?.slug) {
        setTimeout(() => {
          navigate(`/blogs/${data.created[0].meta.slug}`);
        }, 2000);
      } else {
        // Redirect to blogs page after 2 seconds
        setTimeout(() => {
          navigate('/blogs');
        }, 2000);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while generating the blog');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container sx={{ mt: 4, mb: 4, maxWidth: '800px' }}>
      <Paper elevation={2} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Generate Blog
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
          Fill out the form below to generate a new blog post. Only the topic field is required.
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mb: 3 }} onClose={() => setSuccess(null)}>
            {success}
          </Alert>
        )}

        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            {/* Provider Selection */}
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>AI Provider (Optional)</InputLabel>
                <Select
                  value={provider}
                  label="AI Provider (Optional)"
                  onChange={(e) => setProvider(e.target.value)}
                >
                  <MenuItem value="">Default</MenuItem>
                  <MenuItem value="gemini">Gemini</MenuItem>
                  <MenuItem value="openai">OpenAI</MenuItem>
                  <MenuItem value="perplexity">Perplexity</MenuItem>
                </Select>
                <FormHelperText>Select an AI provider or use the default</FormHelperText>
              </FormControl>
            </Grid>

            {/* Topic - Required */}
            <Grid item xs={12}>
              <TextField
                fullWidth
                required
                label="Topic"
                value={formData.topic}
                onChange={handleChange('topic')}
                error={!!errors.topic}
                helperText={errors.topic || 'The main topic of your blog post'}
                placeholder="e.g., Getting Started with React Hooks"
              />
            </Grid>

            {/* Audience */}
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Target Audience"
                value={formData.audience}
                onChange={handleChange('audience')}
                helperText="Who is this blog for?"
                placeholder="e.g., Developers, Beginners, etc."
              />
            </Grid>

            {/* Tone */}
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Tone"
                value={formData.tone}
                onChange={handleChange('tone')}
                helperText="Writing tone and style"
                placeholder="e.g., Clear, friendly, professional"
              />
            </Grid>

            {/* Style */}
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Style"
                value={formData.style}
                onChange={handleChange('style')}
                helperText="Blog post style"
                placeholder="e.g., How-to tutorial, Guide, etc."
              />
            </Grid>

            {/* Length */}
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Length (words)"
                value={formData.length || ''}
                onChange={handleChange('length')}
                error={!!errors.length}
                helperText={errors.length || 'Target word count for the blog'}
                inputProps={{ min: 100 }}
              />
            </Grid>

            {/* Keywords */}
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Keywords (Optional)"
                value={formData.keywords}
                onChange={handleChange('keywords')}
                helperText="Comma-separated keywords for SEO"
                placeholder="e.g., React, Hooks, Tutorial, Guide"
                multiline
                rows={2}
              />
            </Grid>

            {/* Checkboxes */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Content Options
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={formData.include_code}
                      onChange={handleCheckboxChange('include_code')}
                    />
                  }
                  label="Include Code Examples"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={formData.include_tldr}
                      onChange={handleCheckboxChange('include_tldr')}
                    />
                  }
                  label="Include TLDR (Too Long; Didn't Read) Summary"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={formData.include_further_reading}
                      onChange={handleCheckboxChange('include_further_reading')}
                    />
                  }
                  label="Include Further Reading Section"
                />
              </Box>
            </Grid>

            {/* Submit Button */}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                <Button
                  type="button"
                  variant="outlined"
                  onClick={() => navigate('/blogs')}
                  disabled={loading}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="contained"
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} /> : null}
                >
                  {loading ? 'Generating...' : 'Generate Blog'}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </form>
      </Paper>
    </Container>
  );
};

export default GenerateBlog;

