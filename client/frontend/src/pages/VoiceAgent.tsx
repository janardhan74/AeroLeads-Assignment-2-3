import { useState } from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  TextField,
  Button,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Grid,
  Card,
  CardContent,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  Divider,
  IconButton,
  LinearProgress,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PhoneIcon from '@mui/icons-material/Phone';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PhoneCallbackIcon from '@mui/icons-material/PhoneCallback';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`voice-tabpanel-${index}`}
      aria-labelledby={`voice-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

// Make Call Types
interface MakeCallRequest {
  phone_number: string;
  room_name?: string;
  from_number?: string;
  wait_until_answered?: boolean;
  transcript_timeout?: number;
}

interface MakeCallResponse {
  phone_number: string;
  room_name: string;
  dispatch_id: string;
  call_status: {
    outcome: string;
    answered_at: string;
    ended_at: string;
    duration_seconds: number;
    last_status: string;
    call_attributes: Record<string, string>;
  };
  transcript?: {
    transcript_file: string;
    transcript_data: {
      items: Array<{
        id: string;
        type: string;
        role: string;
        content: string[];
        interrupted?: boolean;
      }>;
    };
    items: Array<{
      id: string;
      type: string;
      role: string;
      content: string[];
      interrupted?: boolean;
    }>;
  };
}

// Agent Call Types
interface AgentCallResponse {
  prompt: string;
  response: string;
  success: boolean;
  phone_number?: string;
  room_name?: string;
  dispatch_id?: string;
  call_status?: MakeCallResponse['call_status'];
  transcript?: MakeCallResponse['transcript'];
  // Multiple calls support
  total_calls?: number;
  successful_calls?: number;
  failed_calls?: number;
  call_results?: MakeCallResponse[];
}

// Multiple Calls Types
interface MultipleCallsRequest {
  phone_numbers: string[];
  room_name_prefix?: string;
  from_number?: string;
  wait_until_answered?: boolean;
  transcript_timeout?: number;
  delay_between_calls?: number;
}

interface MultipleCallsResponse {
  total_calls: number;
  successful_calls: number;
  failed_calls: number;
  results: Array<{
    phone_number: string;
    call_index: number;
    start_time: string;
    success: boolean;
    result?: MakeCallResponse;
    error?: string | null;
    end_time: string;
    duration_seconds: number;
  }>;
  start_time: string;
  end_time: string;
  total_duration_seconds: number;
}

const VoiceAgent = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Make Call State
  const [makeCallData, setMakeCallData] = useState<MakeCallRequest>({
    phone_number: '',
    wait_until_answered: true,
    transcript_timeout: 30.0,
  });
  const [makeCallResponse, setMakeCallResponse] = useState<MakeCallResponse | null>(null);

  // Agent Call State
  const [agentPrompt, setAgentPrompt] = useState('');
  const [agentCallResponse, setAgentCallResponse] = useState<AgentCallResponse | null>(null);

  // Multiple Calls State
  const [multipleCallsData, setMultipleCallsData] = useState<MultipleCallsRequest>({
    phone_numbers: [''],
    wait_until_answered: true,
    transcript_timeout: 30.0,
    delay_between_calls: 0.0,
  });
  const [multipleCallsResponse, setMultipleCallsResponse] = useState<MultipleCallsResponse | null>(null);
  const [uploadingFile, setUploadingFile] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    setError(null);
    setUploadError(null);
    setMakeCallResponse(null);
    setAgentCallResponse(null);
    setMultipleCallsResponse(null);
  };

  // Make Call Handlers
  const handleMakeCall = async () => {
    setLoading(true);
    setError(null);
    setMakeCallResponse(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/voice/make-a-call`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(makeCallData),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to make call' }));
        throw new Error(errorData.detail || 'Failed to make call');
      }

      const data: MakeCallResponse = await response.json();
      setMakeCallResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while making the call');
    } finally {
      setLoading(false);
    }
  };

  // Agent Call Handlers
  const handleAgentCall = async () => {
    if (!agentPrompt.trim()) {
      setError('Prompt cannot be empty');
      return;
    }

    setLoading(true);
    setError(null);
    setAgentCallResponse(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/voice/make-agent-call`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: agentPrompt }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to make agent call' }));
        throw new Error(errorData.detail || 'Failed to make agent call');
      }

      const data: AgentCallResponse = await response.json();
      setAgentCallResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while making the agent call');
    } finally {
      setLoading(false);
    }
  };

  // Multiple Calls Handlers
  const handlePhoneNumberChange = (index: number, value: string) => {
    const newNumbers = [...multipleCallsData.phone_numbers];
    newNumbers[index] = value;
    setMultipleCallsData({ ...multipleCallsData, phone_numbers: newNumbers });
  };

  const addPhoneNumber = () => {
    setMultipleCallsData({
      ...multipleCallsData,
      phone_numbers: [...multipleCallsData.phone_numbers, ''],
    });
  };

  const removePhoneNumber = (index: number) => {
    const newNumbers = multipleCallsData.phone_numbers.filter((_, i) => i !== index);
    setMultipleCallsData({ ...multipleCallsData, phone_numbers: newNumbers });
  };

  const parsePhoneNumbersFromFile = (content: string): string[] => {
    // Split by newlines, commas, semicolons, or tabs
    const lines = content.split(/[\n\r,;\t]+/);
    const phoneNumbers: string[] = [];
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) {
        // Extract phone numbers (supports formats like +1234567890, 1234567890, etc.)
        // Remove any non-digit characters except +
        const cleaned = trimmed.replace(/[^\d+]/g, '');
        if (cleaned.length >= 8) { // Minimum phone number length
          // Ensure it starts with + if it doesn't
          const phone = cleaned.startsWith('+') ? cleaned : `+${cleaned}`;
          phoneNumbers.push(phone);
        }
      }
    }
    
    return phoneNumbers;
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadingFile(true);
    setUploadError(null);

    try {
      const text = await file.text();
      const phoneNumbers = parsePhoneNumbersFromFile(text);
      
      if (phoneNumbers.length === 0) {
        setUploadError('No valid phone numbers found in the file. Please check the format.');
      } else {
        // Remove duplicates
        const uniqueNumbers = Array.from(new Set(phoneNumbers));
        setMultipleCallsData({
          ...multipleCallsData,
          phone_numbers: uniqueNumbers.length > 0 ? uniqueNumbers : [''],
        });
        setError(null);
      }
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Failed to read file');
    } finally {
      setUploadingFile(false);
      // Reset file input
      event.target.value = '';
    }
  };

  const clearAllPhoneNumbers = () => {
    setMultipleCallsData({
      ...multipleCallsData,
      phone_numbers: [''],
    });
  };

  const handleMultipleCalls = async () => {
    const validNumbers = multipleCallsData.phone_numbers.filter((num) => num.trim() !== '');
    if (validNumbers.length === 0) {
      setError('At least one phone number is required');
      return;
    }

    setLoading(true);
    setError(null);
    setMultipleCallsResponse(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/voice/make-multiple-calls`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...multipleCallsData,
          phone_numbers: validNumbers,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to make multiple calls' }));
        throw new Error(errorData.detail || 'Failed to make multiple calls');
      }

      const data: MultipleCallsResponse = await response.json();
      setMultipleCallsResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while making multiple calls');
    } finally {
      setLoading(false);
    }
  };

  const renderTranscript = (transcript: MakeCallResponse['transcript']) => {
    if (!transcript || !transcript.items) return null;

    return (
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle2">
            Transcript ({transcript.items.length} items)
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <List dense>
            {transcript.items.map((item, idx) => (
              <Box key={item.id || idx}>
                <ListItem>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip
                          label={item.role}
                          size="small"
                          color={item.role === 'assistant' ? 'primary' : 'default'}
                        />
                        <Typography variant="body2" component="span">
                          {item.type}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <Box sx={{ mt: 1 }}>
                        {item.content.map((text, textIdx) => (
                          <Typography key={textIdx} variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                            {text}
                          </Typography>
                        ))}
                      </Box>
                    }
                  />
                </ListItem>
                {idx < transcript.items.length - 1 && <Divider />}
              </Box>
            ))}
          </List>
        </AccordionDetails>
      </Accordion>
    );
  };

  return (
    <Container sx={{ mt: 4, mb: 4, maxWidth: '1400px' }}>
      <Paper 
        elevation={3} 
        sx={{ 
          p: 4,
          background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
          borderRadius: 3,
        }}
      >
        <Box sx={{ mb: 3 }}>
          <Typography 
            variant="h4" 
            component="h1" 
            gutterBottom
            sx={{ 
              fontWeight: 700,
              background: 'linear-gradient(45deg, #1976d2 30%, #42a5f5 90%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Voice Agent
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
            Make phone calls using the voice agent API. Choose from single calls, agent calls, or batch calls.
          </Typography>
        </Box>

        {error && (
          <Alert 
            severity="error" 
            sx={{ 
              mb: 3,
              borderRadius: 2,
              boxShadow: 2,
            }} 
            onClose={() => setError(null)}
          >
            {error}
          </Alert>
        )}

        <Box 
          sx={{ 
            borderBottom: 2, 
            borderColor: 'divider', 
            mb: 3,
            '& .MuiTabs-indicator': {
              height: 3,
              borderRadius: '3px 3px 0 0',
            },
          }}
        >
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            aria-label="voice agent tabs"
            sx={{
              '& .MuiTab-root': {
                textTransform: 'none',
                fontWeight: 600,
                fontSize: '1rem',
                minHeight: 64,
                '&.Mui-selected': {
                  color: 'primary.main',
                },
              },
            }}
          >
            <Tab icon={<PhoneIcon />} iconPosition="start" label="Make Call" />
            <Tab icon={<SmartToyIcon />} iconPosition="start" label="Agent Call" />
            <Tab icon={<PhoneCallbackIcon />} iconPosition="start" label="Multiple Calls" />
          </Tabs>
        </Box>

        {/* Make Call Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            {/* @ts-expect-error - MUI v7 Grid item prop type issue */}
            <Grid item xs={12} md={6}>
              <Card 
                elevation={4}
                sx={{
                  borderRadius: 3,
                  border: '1px solid',
                  borderColor: 'divider',
                  '& .MuiCardContent-root': {
                    p: 3,
                  },
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <PhoneIcon color="primary" sx={{ fontSize: 28 }} />
                    <Typography 
                      variant="h6" 
                      gutterBottom
                      sx={{ 
                        fontWeight: 600,
                        mb: 0,
                      }}
                    >
                      Make a Call
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5, mt: 2 }}>
                    <TextField
                      label="Phone Number"
                      value={makeCallData.phone_number}
                      onChange={(e) =>
                        setMakeCallData({ ...makeCallData, phone_number: e.target.value })
                      }
                      placeholder="+1234567890"
                      required
                      fullWidth
                    />
                    <TextField
                      label="Room Name (Optional)"
                      value={makeCallData.room_name || ''}
                      onChange={(e) =>
                        setMakeCallData({ ...makeCallData, room_name: e.target.value || undefined })
                      }
                      fullWidth
                    />
                    <TextField
                      label="From Number (Optional)"
                      value={makeCallData.from_number || ''}
                      onChange={(e) =>
                        setMakeCallData({ ...makeCallData, from_number: e.target.value || undefined })
                      }
                      fullWidth
                    />
                    <TextField
                      label="Transcript Timeout (seconds)"
                      type="number"
                      value={makeCallData.transcript_timeout}
                      onChange={(e) =>
                        setMakeCallData({
                          ...makeCallData,
                          transcript_timeout: parseFloat(e.target.value) || 30.0,
                        })
                      }
                      fullWidth
                    />
                    <Button
                      variant="contained"
                      onClick={handleMakeCall}
                      disabled={loading || !makeCallData.phone_number.trim()}
                      startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <PhoneIcon />}
                      fullWidth
                      sx={{
                        py: 1.5,
                        fontSize: '1rem',
                        fontWeight: 600,
                        borderRadius: 2,
                        textTransform: 'none',
                        boxShadow: 3,
                        '&:hover': {
                          boxShadow: 4,
                          transform: 'translateY(-2px)',
                          transition: 'all 0.2s',
                        },
                      }}
                    >
                      {loading ? 'Calling...' : 'Make Call'}
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            {/* @ts-expect-error - MUI v7 Grid item prop type issue */}
            <Grid item xs={12} md={6}>
              {makeCallResponse && (
                <Card 
                  elevation={4}
                  sx={{
                    borderRadius: 3,
                    border: '1px solid',
                    borderColor: 'divider',
                    '& .MuiCardContent-root': {
                      p: 3,
                    },
                  }}
                >
                  <CardContent>
                    <Typography 
                      variant="h6" 
                      gutterBottom
                      sx={{ 
                        fontWeight: 600,
                        mb: 2,
                        color: 'primary.main',
                      }}
                    >
                      Call Result
                    </Typography>
                    <Box sx={{ mt: 2 }}>
                      <Box sx={{ 
                        display: 'flex', 
                        flexDirection: 'column', 
                        gap: 1.5,
                        p: 2,
                        backgroundColor: 'grey.50',
                        borderRadius: 2,
                        mb: 2,
                      }}>
                        <Typography variant="body2">
                          <strong>Phone:</strong> {makeCallResponse.phone_number}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Room:</strong> {makeCallResponse.room_name}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Dispatch ID:</strong> {makeCallResponse.dispatch_id}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                          <Typography variant="body2" component="span">
                            <strong>Status:</strong>
                          </Typography>
                          <Chip
                            label={makeCallResponse.call_status.outcome}
                            size="small"
                            color={
                              makeCallResponse.call_status.outcome === 'completed'
                                ? 'success'
                                : 'default'
                            }
                            sx={{ fontWeight: 600 }}
                          />
                        </Box>
                        <Typography variant="body2">
                          <strong>Duration:</strong> {makeCallResponse.call_status.duration_seconds} seconds
                        </Typography>
                      </Box>
                      {makeCallResponse.transcript && renderTranscript(makeCallResponse.transcript)}
                    </Box>
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </TabPanel>

        {/* Agent Call Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            {/* @ts-expect-error - MUI v7 Grid item prop type issue */}
            <Grid item xs={12} md={8}>
              <Card 
                elevation={4}
                sx={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  borderRadius: 3,
                  '& .MuiCardContent-root': {
                    p: 3,
                  },
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                    <SmartToyIcon sx={{ color: 'white', fontSize: 28 }} />
                    <Typography 
                      variant="h5" 
                      gutterBottom
                      sx={{ 
                        color: 'white',
                        fontWeight: 600,
                      }}
                    >
                      Agent Call
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <TextField
                      label="Prompt"
                      value={agentPrompt}
                      onChange={(e) => setAgentPrompt(e.target.value)}
                      placeholder="make call to +1234567890 and +9876543210"
                      required
                      fullWidth
                      multiline
                      rows={6}
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          backgroundColor: 'white',
                          borderRadius: 2,
                          '& fieldset': {
                            borderColor: 'rgba(255, 255, 255, 0.5)',
                          },
                          '&:hover fieldset': {
                            borderColor: 'rgba(255, 255, 255, 0.8)',
                          },
                          '&.Mui-focused fieldset': {
                            borderColor: 'white',
                            borderWidth: 2,
                          },
                        },
                        '& .MuiInputLabel-root': {
                          color: 'rgba(255, 255, 255, 0.9)',
                        },
                        '& .MuiInputLabel-root.Mui-focused': {
                          color: 'white',
                        },
                      }}
                    />
                    <Button
                      variant="contained"
                      onClick={handleAgentCall}
                      disabled={loading || !agentPrompt.trim()}
                      startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SmartToyIcon />}
                      fullWidth
                      sx={{
                        py: 1.5,
                        fontSize: '1rem',
                        fontWeight: 600,
                        borderRadius: 2,
                        background: 'white',
                        color: '#667eea',
                        textTransform: 'none',
                        boxShadow: 3,
                        '&:hover': {
                          background: 'rgba(255, 255, 255, 0.9)',
                          boxShadow: 4,
                          transform: 'translateY(-2px)',
                          transition: 'all 0.2s',
                        },
                        '&:disabled': {
                          background: 'rgba(255, 255, 255, 0.5)',
                          color: 'rgba(102, 126, 234, 0.5)',
                        },
                      }}
                    >
                      {loading ? 'Calling...' : 'Make Agent Call'}
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            {/* @ts-expect-error - MUI v7 Grid item prop type issue */}
            <Grid item xs={12} md={4}>
              {agentCallResponse && (
                <Card 
                  elevation={4}
                  sx={{
                    borderRadius: 3,
                    border: '1px solid',
                    borderColor: 'divider',
                    '& .MuiCardContent-root': {
                      p: 3,
                    },
                  }}
                >
                  <CardContent>
                    <Typography 
                      variant="h6" 
                      gutterBottom
                      sx={{ 
                        fontWeight: 600,
                        mb: 2,
                        color: 'primary.main',
                      }}
                    >
                      Agent Response
                    </Typography>
                    <Box sx={{ mt: 2 }}>
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          mb: 2,
                          backgroundColor: 'grey.50',
                          borderRadius: 2,
                          border: '1px solid',
                          borderColor: 'grey.200',
                        }}
                      >
                        <Typography 
                          variant="body1" 
                          sx={{ 
                            whiteSpace: 'pre-wrap',
                            lineHeight: 1.7,
                            color: 'text.primary',
                          }}
                        >
                          {agentCallResponse.response}
                        </Typography>
                      </Paper>
                      {/* Multiple Calls Results */}
                      {agentCallResponse.call_results && agentCallResponse.call_results.length > 1 && (
                        <>
                          <Box sx={{ 
                            display: 'flex', 
                            flexDirection: 'column', 
                            gap: 1.5,
                            p: 2,
                            backgroundColor: 'grey.50',
                            borderRadius: 2,
                            mb: 2,
                          }}>
                            <Typography variant="body2">
                              <strong>Total Calls:</strong> {agentCallResponse.total_calls || agentCallResponse.call_results.length}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="body2" component="span">
                                <strong>Successful:</strong>
                              </Typography>
                              <Chip
                                label={agentCallResponse.successful_calls || 0}
                                size="small"
                                color="success"
                                sx={{ fontWeight: 600 }}
                              />
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="body2" component="span">
                                <strong>Failed:</strong>
                              </Typography>
                              <Chip
                                label={agentCallResponse.failed_calls || 0}
                                size="small"
                                color="error"
                                sx={{ fontWeight: 600 }}
                              />
                            </Box>
                          </Box>
                          <Divider sx={{ my: 2 }} />
                          <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                            Individual Call Results:
                          </Typography>
                          {agentCallResponse.call_results.map((result, idx) => (
                            <Accordion key={idx} sx={{ mt: 1 }}>
                              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                                  <Chip
                                    label={result.call_status?.outcome || 'unknown'}
                                    size="small"
                                    color={
                                      result.call_status?.outcome === 'completed'
                                        ? 'success'
                                        : 'default'
                                    }
                                  />
                                  <Typography variant="body2">{result.phone_number}</Typography>
                                  {result.call_status && (
                                    <Typography variant="body2" sx={{ ml: 'auto' }}>
                                      {result.call_status.duration_seconds}s
                                    </Typography>
                                  )}
                                </Box>
                              </AccordionSummary>
                              <AccordionDetails>
                                <Box>
                                  <Typography variant="body2">
                                    <strong>Room:</strong> {result.room_name}
                                  </Typography>
                                  {result.call_status && (
                                    <>
                                      <Typography variant="body2" sx={{ mt: 0.5 }}>
                                        <strong>Status:</strong>{' '}
                                        <Chip
                                          label={result.call_status.outcome}
                                          size="small"
                                          color={
                                            result.call_status.outcome === 'completed'
                                              ? 'success'
                                              : 'default'
                                          }
                                          sx={{ fontWeight: 600 }}
                                        />
                                      </Typography>
                                      <Typography variant="body2">
                                        <strong>Duration:</strong>{' '}
                                        {result.call_status.duration_seconds} seconds
                                      </Typography>
                                    </>
                                  )}
                                  {result.transcript && renderTranscript(result.transcript)}
                                </Box>
                              </AccordionDetails>
                            </Accordion>
                          ))}
                        </>
                      )}
                      {/* Single Call Result */}
                      {agentCallResponse.phone_number && !agentCallResponse.call_results && (
                        <>
                          <Box sx={{ 
                            display: 'flex', 
                            flexDirection: 'column', 
                            gap: 1.5,
                            p: 2,
                            backgroundColor: 'grey.50',
                            borderRadius: 2,
                            mb: 2,
                          }}>
                            <Typography variant="body2">
                              <strong>Phone:</strong> {agentCallResponse.phone_number}
                            </Typography>
                            <Typography variant="body2">
                              <strong>Room:</strong> {agentCallResponse.room_name}
                            </Typography>
                            {agentCallResponse.call_status && (
                              <>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                                  <Typography variant="body2" component="span">
                                    <strong>Status:</strong>
                                  </Typography>
                                  <Chip
                                    label={agentCallResponse.call_status.outcome}
                                    size="small"
                                    color={
                                      agentCallResponse.call_status.outcome === 'completed'
                                        ? 'success'
                                        : 'default'
                                    }
                                    sx={{ fontWeight: 600 }}
                                  />
                                </Box>
                                <Typography variant="body2">
                                  <strong>Duration:</strong>{' '}
                                  {agentCallResponse.call_status.duration_seconds} seconds
                                </Typography>
                              </>
                            )}
                          </Box>
                          {agentCallResponse.transcript && renderTranscript(agentCallResponse.transcript)}
                        </>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </TabPanel>

        {/* Multiple Calls Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            {/* @ts-expect-error - MUI v7 Grid item prop type issue */}
            <Grid item xs={12} md={6}>
              <Card 
                elevation={4}
                sx={{
                  borderRadius: 3,
                  border: '1px solid',
                  borderColor: 'divider',
                  '& .MuiCardContent-root': {
                    p: 3,
                  },
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <PhoneCallbackIcon color="primary" sx={{ fontSize: 28 }} />
                    <Typography 
                      variant="h6" 
                      gutterBottom
                      sx={{ 
                        fontWeight: 600,
                        mb: 0,
                      }}
                    >
                      Multiple Calls
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5, mt: 2 }}>
                    {/* File Upload Section */}
                    <Paper
                      elevation={0}
                      sx={{
                        p: 3,
                        border: '2px dashed',
                        borderColor: 'primary.main',
                        borderRadius: 2,
                        backgroundColor: 'grey.50',
                        textAlign: 'center',
                        transition: 'all 0.3s',
                        '&:hover': {
                          borderColor: 'primary.dark',
                          backgroundColor: 'grey.100',
                        },
                      }}
                    >
                      <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
                      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                        Upload Phone Numbers File
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Upload a file with phone numbers (one per line, or comma/tab separated)
                        <br />
                        Supported formats: .txt, .csv
                      </Typography>
                      {uploadingFile && (
                        <Box sx={{ width: '100%', mb: 2 }}>
                          <LinearProgress />
                        </Box>
                      )}
                      {uploadError && (
                        <Alert severity="error" sx={{ mb: 2, textAlign: 'left' }}>
                          {uploadError}
                        </Alert>
                      )}
                      <input
                        accept=".txt,.csv"
                        style={{ display: 'none' }}
                        id="phone-numbers-file-upload"
                        type="file"
                        onChange={handleFileUpload}
                        disabled={uploadingFile}
                      />
                      <label htmlFor="phone-numbers-file-upload">
                        <Button
                          variant="contained"
                          component="span"
                          startIcon={<CloudUploadIcon />}
                          disabled={uploadingFile}
                          sx={{
                            textTransform: 'none',
                            fontWeight: 600,
                            borderRadius: 2,
                            px: 3,
                          }}
                        >
                          {uploadingFile ? 'Uploading...' : 'Choose File'}
                        </Button>
                      </label>
                      {multipleCallsData.phone_numbers.length > 0 && 
                       multipleCallsData.phone_numbers.some(num => num.trim() !== '') && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="body2" color="success.main" sx={{ fontWeight: 600 }}>
                            {multipleCallsData.phone_numbers.filter(num => num.trim() !== '').length} phone number(s) loaded
                          </Typography>
                          <Button
                            variant="outlined"
                            color="error"
                            size="small"
                            startIcon={<DeleteIcon />}
                            onClick={clearAllPhoneNumbers}
                            sx={{ mt: 1, textTransform: 'none' }}
                          >
                            Clear All
                          </Button>
                        </Box>
                      )}
                    </Paper>

                    <Divider sx={{ my: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        OR Enter Manually
                      </Typography>
                    </Divider>

                    {/* Manual Entry Section */}
                    {multipleCallsData.phone_numbers.map((phone, index) => (
                      <Box key={index} sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                        <TextField
                          label={`Phone Number ${index + 1}`}
                          value={phone}
                          onChange={(e) => handlePhoneNumberChange(index, e.target.value)}
                          placeholder="+1234567890"
                          fullWidth
                        />
                        {multipleCallsData.phone_numbers.length > 1 && (
                          <IconButton
                            color="error"
                            onClick={() => removePhoneNumber(index)}
                            sx={{ 
                              minWidth: 'auto',
                              '&:hover': {
                                backgroundColor: 'error.light',
                                color: 'white',
                              },
                            }}
                          >
                            <DeleteIcon />
                          </IconButton>
                        )}
                      </Box>
                    ))}
                    <Button 
                      variant="outlined" 
                      onClick={addPhoneNumber} 
                      fullWidth
                      sx={{
                        py: 1.2,
                        borderRadius: 2,
                        textTransform: 'none',
                        fontWeight: 600,
                        borderWidth: 2,
                        '&:hover': {
                          borderWidth: 2,
                        },
                      }}
                    >
                      Add Phone Number
                    </Button>
                    <TextField
                      label="Room Name Prefix (Optional)"
                      value={multipleCallsData.room_name_prefix || ''}
                      onChange={(e) =>
                        setMultipleCallsData({
                          ...multipleCallsData,
                          room_name_prefix: e.target.value || undefined,
                        })
                      }
                      fullWidth
                    />
                    <TextField
                      label="Delay Between Calls (seconds)"
                      type="number"
                      value={multipleCallsData.delay_between_calls}
                      onChange={(e) =>
                        setMultipleCallsData({
                          ...multipleCallsData,
                          delay_between_calls: parseFloat(e.target.value) || 0.0,
                        })
                      }
                      fullWidth
                    />
                    <Button
                      variant="contained"
                      onClick={handleMultipleCalls}
                      disabled={
                        loading ||
                        multipleCallsData.phone_numbers.filter((num) => num.trim() !== '').length === 0
                      }
                      startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <PhoneCallbackIcon />}
                      fullWidth
                      sx={{
                        py: 1.5,
                        fontSize: '1rem',
                        fontWeight: 600,
                        borderRadius: 2,
                        textTransform: 'none',
                        boxShadow: 3,
                        '&:hover': {
                          boxShadow: 4,
                          transform: 'translateY(-2px)',
                          transition: 'all 0.2s',
                        },
                      }}
                    >
                      {loading ? 'Calling...' : 'Make Multiple Calls'}
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            {/* @ts-expect-error - MUI v7 Grid item prop type issue */}
            <Grid item xs={12} md={6}>
              {multipleCallsResponse && (
                <Card 
                  elevation={4}
                  sx={{
                    borderRadius: 3,
                    border: '1px solid',
                    borderColor: 'divider',
                    '& .MuiCardContent-root': {
                      p: 3,
                    },
                  }}
                >
                  <CardContent>
                    <Typography 
                      variant="h6" 
                      gutterBottom
                      sx={{ 
                        fontWeight: 600,
                        mb: 2,
                        color: 'primary.main',
                      }}
                    >
                      Batch Call Results
                    </Typography>
                    <Box sx={{ mt: 2 }}>
                      <Box sx={{ 
                        display: 'flex', 
                        flexDirection: 'column', 
                        gap: 1.5,
                        p: 2,
                        backgroundColor: 'grey.50',
                        borderRadius: 2,
                        mb: 2,
                      }}>
                        <Typography variant="body2">
                          <strong>Total Calls:</strong> {multipleCallsResponse.total_calls}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2" component="span">
                            <strong>Successful:</strong>
                          </Typography>
                          <Chip
                            label={multipleCallsResponse.successful_calls}
                            size="small"
                            color="success"
                            sx={{ fontWeight: 600 }}
                          />
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2" component="span">
                            <strong>Failed:</strong>
                          </Typography>
                          <Chip
                            label={multipleCallsResponse.failed_calls}
                            size="small"
                            color="error"
                            sx={{ fontWeight: 600 }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ mt: 0.5 }}>
                          <strong>Total Duration:</strong>{' '}
                          {multipleCallsResponse.total_duration_seconds.toFixed(2)} seconds
                        </Typography>
                      </Box>
                      <Divider sx={{ my: 2 }} />
                      <Typography variant="subtitle2" gutterBottom>
                        Individual Results:
                      </Typography>
                      {multipleCallsResponse.results.map((result, idx) => (
                        <Accordion key={idx} sx={{ mt: 1 }}>
                          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                              <Chip
                                label={result.success ? 'Success' : 'Failed'}
                                size="small"
                                color={result.success ? 'success' : 'error'}
                              />
                              <Typography variant="body2">{result.phone_number}</Typography>
                              <Typography variant="body2" sx={{ ml: 'auto' }}>
                                {result.duration_seconds.toFixed(2)}s
                              </Typography>
                            </Box>
                          </AccordionSummary>
                          <AccordionDetails>
                            {result.success && result.result && (
                              <Box>
                                <Typography variant="body2">
                                  <strong>Status:</strong>{' '}
                                  <Chip
                                    label={result.result.call_status.outcome}
                                    size="small"
                                    color={
                                      result.result.call_status.outcome === 'completed'
                                        ? 'success'
                                        : 'default'
                                    }
                                  />
                                </Typography>
                                <Typography variant="body2">
                                  <strong>Duration:</strong>{' '}
                                  {result.result.call_status.duration_seconds} seconds
                                </Typography>
                                {result.result.transcript && renderTranscript(result.result.transcript)}
                              </Box>
                            )}
                            {!result.success && result.error && (
                              <Typography variant="body2" color="error">
                                Error: {result.error}
                              </Typography>
                            )}
                          </AccordionDetails>
                        </Accordion>
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default VoiceAgent;

