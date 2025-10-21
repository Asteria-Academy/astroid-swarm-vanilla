/**
 * Agents JavaScript file
 * Handles all agent-related functionality
 */

// Global variables
let currentAgentId = null;
let allTools = [];
let allCompanies = [];

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Check authentication
    if (!Utils.checkAuth()) return;
    
    // Load agents
    loadAgents();
    
    // Load companies for filter and form
    loadCompanies();
    
    // Load tools for form
    loadTools();
    
    // Event listeners
    document.getElementById('refresh-agents').addEventListener('click', loadAgents);
    document.getElementById('company-filter').addEventListener('change', loadAgents);
    document.getElementById('save-agent').addEventListener('click', saveAgent);
    document.getElementById('add-tool-btn').addEventListener('click', showAddToolModal);
    document.getElementById('confirm-add-tool').addEventListener('click', addToolToAgent);
    // autofill-style-btn now uses inline onclick handler
    
    // Clone agent event listeners
    document.getElementById('clone-agent-btn').addEventListener('click', showCloneAgentModal);
    document.getElementById('confirm-clone').addEventListener('click', cloneSelectedAgent);
    
    // Reset form when modal is opened for creating a new agent
    document.getElementById('create-agent-btn').addEventListener('click', function() {
        resetAgentForm();
        document.getElementById('agent-modal-label').textContent = 'Create Agent';
    });

    // Reset the company dropdown to "Personal Agent" (the first option)
    document.getElementById('agent-modal').addEventListener('show.bs.modal', function (event) {
        const companySelect = document.getElementById('agent-company');
        companySelect.selectedIndex = 0;
        loadTools();
    });
});

// Load all agents
async function loadAgents() {
    try {
        Utils.showLoading('agents-container');
        
        const companyId = document.getElementById('company-filter').value;
        let endpoint = '/agents';
        
        if (companyId) {
            endpoint += `?company_id=${companyId}`;
        }
        
        const agents = await API.get(endpoint);
        
        if (agents.length === 0) {
            Utils.hideLoading('agents-container', '<p class="text-center">No agents found</p>');
            return;
        }
        
        let html = '<div class="row">';
        
        agents.forEach(agent => {
            html += `
                <div class="col-md-4 mb-3">
                    <div class="card agent-card h-100">
                        <div class="card-body">
                            <h5 class="card-title">${agent.agent_name}</h5>
                            <p class="card-text">${agent.description || 'No description'}</p>
                            <p class="mb-1"><small class="text-muted">Status: 
                                <span class="badge ${agent.on_status ? 'bg-success' : 'bg-danger'}">
                                    ${agent.on_status ? 'Active' : 'Inactive'}
                                </span>
                            </small></p>
                            <p class="mb-1"><small class="text-muted">Style: ${agent.agent_style || 'Default'}</small></p>
                            <p class="mb-1"><small class="text-muted">Company: ${agent.company_id ? 'Company Agent' : 'Personal Agent'}</small></p>
                        </div>
                        <div class="card-footer">
                            <button class="btn btn-sm btn-primary view-agent" data-id="${agent.agent_id}">View</button>
                            <button class="btn btn-sm btn-info edit-agent" data-id="${agent.agent_id}">Edit</button>
                            <button class="btn btn-sm btn-danger delete-agent" data-id="${agent.agent_id}">Delete</button>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        
        Utils.hideLoading('agents-container', html);
        
        // Add event listeners to buttons
        document.querySelectorAll('.view-agent').forEach(button => {
            button.addEventListener('click', function() {
                const agentId = this.getAttribute('data-id');
                loadAgentDetails(agentId);
            });
        });
        
        document.querySelectorAll('.edit-agent').forEach(button => {
            button.addEventListener('click', function() {
                const agentId = this.getAttribute('data-id');
                editAgent(agentId);
            });
        });
        
        document.querySelectorAll('.delete-agent').forEach(button => {
            button.addEventListener('click', function() {
                const agentId = this.getAttribute('data-id');
                deleteAgent(agentId);
            });
        });
        
    } catch (error) {
        Utils.hideLoading('agents-container', `<p class="text-center text-danger">Error loading agents: ${error.detail || error.message || 'Unknown error'}</p>`);
    }
}

// Load agent details
async function loadAgentDetails(agentId) {
    try {
        currentAgentId = agentId;
        
        Utils.showLoading('agent-details-container');
        Utils.showLoading('agent-tools-container');
        
        // Enable the add tool button
        document.getElementById('add-tool-btn').disabled = false;
        
        const agent = await API.get(`/agents/${agentId}`);
        
        let detailsHtml = `
            <h4>${agent.agent_name}</h4>
            <p>${agent.description || 'No description'}</p>
            <div class="row">
                <div class="col-md-6">
                    <p><strong>Agent ID:</strong> ${agent.agent_id}</p>
                    <p><strong>User ID:</strong> ${agent.user_id}</p>
                    <p><strong>Status:</strong> 
                        <span class="badge ${agent.on_status ? 'bg-success' : 'bg-danger'}">
                            ${agent.on_status ? 'Active' : 'Inactive'}
                        </span>
                    </p>
                </div>
                <div class="col-md-6">
                    <p><strong>Style:</strong> ${agent.agent_style || 'Default'}</p>
                    <p><strong>Company ID:</strong> ${agent.company_id || 'Personal Agent'}</p>
                </div>
            </div>
            <p><strong>Created:</strong> ${Utils.formatDate(agent.created_at)}</p>
        `;
        
        Utils.hideLoading('agent-details-container', detailsHtml);
        
        // Load agent tools
        loadAgentTools(agentId, agent.tool_details);
        
    } catch (error) {
        Utils.hideLoading('agent-details-container', `<p class="text-center text-danger">Error loading agent details: ${error.detail || error.message || 'Unknown error'}</p>`);
        Utils.hideLoading('agent-tools-container', '<p class="text-center">Failed to load tools</p>');
    }
}

// Load agent tools
function loadAgentTools(agentId, tools) {
    if (!tools || tools.length === 0) {
        Utils.hideLoading('agent-tools-container', '<p class="text-center">No tools assigned to this agent</p>');
        return;
    }
    
    let toolsHtml = '<div class="row">';
    
    tools.forEach(tool => {
        toolsHtml += `
            <div class="col-md-4 mb-3">
                <div class="card tool-card h-100">
                    <div class="card-body">
                        <h5 class="card-title">${tool.name}</h5>
                        <p class="card-text">${tool.description || 'No description'}</p>
                        <p class="mb-1"><small class="text-muted">Tool ID: ${tool.tool_id}</small></p>
                        <p class="mb-0"><small class="text-muted">Versions: ${formatVersions(tool.versions)}</small></p>
                    </div>
                    <div class="card-footer">
                        <button class="btn btn-sm btn-danger remove-tool" data-tool-id="${tool.tool_id}">Remove</button>
                    </div>
                </div>
            </div>
        `;
    });
    
    toolsHtml += '</div>';
    
    Utils.hideLoading('agent-tools-container', toolsHtml);
    
    // Add event listeners to remove tool buttons
    document.querySelectorAll('.remove-tool').forEach(button => {
        button.addEventListener('click', function() {
            const toolId = this.getAttribute('data-tool-id');
            removeToolFromAgent(toolId);
        });
    });
}

// Format tool versions
function formatVersions(versions) {
    if (!versions || versions.length === 0) {
        return 'None';
    }
    
    return versions.map(v => v.version).join(', ');
}

// Load companies for filter and form
async function loadCompanies() {
    try {
        const companies = await API.get('/companies');
        allCompanies = companies;
        
        const filterSelect = document.getElementById('company-filter');
        const formSelect = document.getElementById('agent-company');
        
        // Clear existing options (except the first one)
        while (filterSelect.options.length > 1) {
            filterSelect.remove(1);
        }
        
        while (formSelect.options.length > 1) {
            formSelect.remove(1);
        }
        
        // Add companies to selects
        companies.forEach(company => {
            const filterOption = new Option(company.name, company.company_id);
            const formOption = new Option(company.name, company.company_id);
            
            filterSelect.add(filterOption);
            formSelect.add(formOption);
        });
        
    } catch (error) {
        console.error('Error loading companies:', error);
    }
}

document.getElementById('agent-company').addEventListener('change', async function () {
    const companyId = this.value;
    await loadTools(companyId);
});

// Load tools for form
async function loadTools(companyId = null) {
    try {
        let url = '/tools';
        if (companyId && companyId !== '') {
            url += `?company_id=${companyId}`;
        }

        const tools = await API.get(url);
        allTools = tools;
        
        const toolsSelect = document.getElementById('agent-tools');
        
        // Clear existing options
        toolsSelect.innerHTML = '';
        
        // Add tools to select
        tools.forEach(tool => {
            const option = new Option(tool.name, tool.tool_id);
            toolsSelect.add(option);
        });
        
    } catch (error) {
        console.error('Error loading tools:', error);
    }
}

// Reset agent form
function resetAgentForm() {
    document.getElementById('agent-form').reset();
    document.getElementById('agent-id').value = '';
    
    // Clear selected tools
    const toolsSelect = document.getElementById('agent-tools');
    for (let i = 0; i < toolsSelect.options.length; i++) {
        toolsSelect.options[i].selected = false;
    }
}

// Save agent (create or update)
async function saveAgent() {
    try {
        const agentId = document.getElementById('agent-id').value;
        const isUpdate = !!agentId;
        
        // Get form values
        const agentName = document.getElementById('agent-name').value;
        const description = document.getElementById('agent-description').value;
        const agentStyleText = document.getElementById('agent-style').value;
        const onStatus = document.getElementById('agent-status').checked;
        const companyId = document.getElementById('agent-company').value;
        
        // Use agent style as normal text, with default if empty
        let agentStyle = agentStyleText.trim();
        if (!agentStyle) {
            // Use default text if the field is empty
            agentStyle = "The agent will reply in a warm and friendly manner, using English.";
        }
        console.log('Using agent style as text:', agentStyle);
        
        // Get selected tools
        const toolsSelect = document.getElementById('agent-tools');
        const selectedTools = Array.from(toolsSelect.selectedOptions).map(option => option.value);
        
        // Create agent data object
        const agentData = {
            agent_name: agentName,
            description: description,
            agent_style: agentStyle,
            on_status: onStatus,
            tools: selectedTools
        };
        
        // Add company_id if selected
        if (companyId) {
            agentData.company_id = companyId;
        }
        
        let response;
        
        if (isUpdate) {
            // Update existing agent
            response = await API.put(`/agents/${agentId}`, agentData);
            Utils.showNotification('Agent updated successfully');
        } else {
            // Create new agent
            response = await API.post('/agents', agentData);
            Utils.showNotification('Agent created successfully');
        }
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('agent-modal'));
        modal.hide();
        
        // Reload agents
        loadAgents();
        
        // If we were viewing the agent that was updated, reload its details
        if (isUpdate && currentAgentId === agentId) {
            loadAgentDetails(agentId);
        }
        
    } catch (error) {
        Utils.showNotification(`Error saving agent: ${error.detail || error.message || 'Unknown error'}`, 'danger');
    }
}

// Edit agent
async function editAgent(agentId) {
    try {
        const agent = await API.get(`/agents/${agentId}`);
        
        // Set form values
        document.getElementById('agent-id').value = agent.agent_id;
        document.getElementById('agent-name').value = agent.agent_name;
        document.getElementById('agent-description').value = agent.description || '';
        
        // Handle agent style as normal text
        const styleInput = document.getElementById('agent-style');
        if (agent.agent_style) {
            // For backward compatibility, handle both string and object types
            if (typeof agent.agent_style === 'object') {
                // If it's somehow stored as an object, convert to a descriptive string
                try {
                    styleInput.value = JSON.stringify(agent.agent_style);
                } catch (e) {
                    styleInput.value = "The agent will reply in a warm and friendly manner, using English.";
                }
            } else {
                // Use the text as-is
                styleInput.value = agent.agent_style;
            }
        } else {
            // Default text if no style is defined
            styleInput.value = "The agent will reply in a warm and friendly manner, using English.";
        }
        
        document.getElementById('agent-status').checked = agent.on_status;
        document.getElementById('agent-company').value = agent.company_id || '';
        
        // Set selected tools
        const toolsSelect = document.getElementById('agent-tools');
        for (let i = 0; i < toolsSelect.options.length; i++) {
            const option = toolsSelect.options[i];
            option.selected = agent.tools && agent.tools.includes(option.value);
        }
        
        // Update modal title
        document.getElementById('agent-modal-label').textContent = 'Edit Agent';
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('agent-modal'));
        modal.show();
        
    } catch (error) {
        Utils.showNotification(`Error loading agent for editing: ${error.detail || error.message || 'Unknown error'}`, 'danger');
    }
}

// Delete agent
async function deleteAgent(agentId) {
    if (!confirm('Are you sure you want to delete this agent?')) {
        return;
    }
    
    try {
        await API.delete(`/agents/${agentId}`);
        Utils.showNotification('Agent deleted successfully');
        
        // Reload agents
        loadAgents();
        
        // If we were viewing the agent that was deleted, clear the details
        if (currentAgentId === agentId) {
            currentAgentId = null;
            document.getElementById('agent-details-container').innerHTML = '<p class="text-center">Select an agent to view details</p>';
            document.getElementById('agent-tools-container').innerHTML = '<p class="text-center">Select an agent to view its tools</p>';
            document.getElementById('add-tool-btn').disabled = true;
        }
        
    } catch (error) {
        Utils.showNotification(`Error deleting agent: ${error.detail || error.message || 'Unknown error'}`, 'danger');
    }
}

// Show add tool modal
function showAddToolModal() {
    if (!currentAgentId) {
        Utils.showNotification('Please select an agent first', 'warning');
        return;
    }
    
    // Get current agent tools
    const agentToolsContainer = document.getElementById('agent-tools-container');
    const toolElements = agentToolsContainer.querySelectorAll('.remove-tool');
    const currentToolIds = Array.from(toolElements).map(el => el.getAttribute('data-tool-id'));
    
    // Filter out tools that are already assigned to the agent
    const availableTools = allTools.filter(tool => !currentToolIds.includes(tool.tool_id));
    
    const availableToolsSelect = document.getElementById('available-tools');
    availableToolsSelect.innerHTML = '';
    
    if (availableTools.length === 0) {
        availableToolsSelect.innerHTML = '<option value="">No available tools</option>';
        document.getElementById('confirm-add-tool').disabled = true;
    } else {
        availableTools.forEach(tool => {
            const option = new Option(tool.name, tool.tool_id);
            availableToolsSelect.add(option);
        });
        document.getElementById('confirm-add-tool').disabled = false;
    }
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('add-tool-modal'));
    modal.show();
}

// Add tool to agent
async function addToolToAgent() {
    if (!currentAgentId) {
        Utils.showNotification('Please select an agent first', 'warning');
        return;
    }
    
    const toolId = document.getElementById('available-tools').value;
    
    if (!toolId) {
        Utils.showNotification('Please select a tool', 'warning');
        return;
    }
    
    try {
        await API.post(`/agents/${currentAgentId}/tools/${toolId}`);
        Utils.showNotification('Tool added to agent successfully');
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('add-tool-modal'));
        modal.hide();
        
        // Reload agent details
        loadAgentDetails(currentAgentId);
        
    } catch (error) {
        Utils.showNotification(`Error adding tool to agent: ${error.detail || error.message || 'Unknown error'}`, 'danger');
    }
}

// Remove tool from agent
async function removeToolFromAgent(toolId) {
    if (!currentAgentId) {
        Utils.showNotification('Please select an agent first', 'warning');
        return;
    }
    
    if (!confirm('Are you sure you want to remove this tool from the agent?')) {
        return;
    }
    
    try {
        await API.delete(`/agents/${currentAgentId}/tools/${toolId}`);
        Utils.showNotification('Tool removed from agent successfully');
        
        // Reload agent details
        loadAgentDetails(currentAgentId);
        
    } catch (error) {
        Utils.showNotification(`Error removing tool from agent: ${error.detail || error.message || 'Unknown error'}`, 'danger');
    }
}

// Autofill agent style using the agent_field_autofill API
async function autofillAgentStyle() {
    try {
        // Get current form values to use for autofill
        const agentName = document.getElementById('agent-name').value;
        const description = document.getElementById('agent-description').value;
        
        // Check if we have enough information to generate a style
        if (!agentName) {
            Utils.showNotification('Please enter an agent name first', 'warning');
            return;
        }
        
        // Prepare the JSON field data
        const jsonField = {
            agent_name: agentName
        };
        
        // Add description if available
        if (description) {
            jsonField.description = description;
        }
        
        // Get the style textarea and its current value
        const styleTextarea = document.getElementById('agent-style');
        const originalValue = styleTextarea.value;
        
        // Disable the textarea during generation
        styleTextarea.disabled = true;
        
        try {
            // Use the API utility for consistency
            const response = await API.post('/agent-field-autofill/invoke', {
                field_name: "agent_style",
                json_field: jsonField,
                existing_field_value: originalValue
            });
            
            // Update the style textarea with the autofilled value
            if (response && response.autofilled_value) {
                // Simulate streaming by adding characters one by one
                let displayedText = originalValue || "";
                const newText = response.autofilled_value;
                
                // Function to add one character at a time
                const typeText = async (text, index) => {
                    if (index < text.length) {
                        displayedText += text[index];
                        styleTextarea.value = displayedText + "▌"; // Add cursor indicator
                        
                        // Auto-scroll to the bottom
                        styleTextarea.scrollTop = styleTextarea.scrollHeight;
                        
                        // Wait a small random amount of time before adding the next character
                        const delay = Math.floor(Math.random() * 30) + 10; // 10-40ms
                        await new Promise(resolve => setTimeout(resolve, delay));
                        
                        // Continue with the next character
                        await typeText(text, index + 1);
                    } else {
                        // Finished typing
                        styleTextarea.value = displayedText;
                        Utils.showNotification('Agent style autofilled successfully');
                    }
                };
                
                // Start typing the new text
                await typeText(newText.substring(originalValue ? originalValue.length : 0), 0);
            } else {
                // Restore original value if no result
                styleTextarea.value = originalValue;
                Utils.showNotification('Failed to autofill agent style', 'warning');
            }
        } catch (error) {
            // Restore original value and show error
            styleTextarea.value = originalValue;
            Utils.showNotification(`Error autofilling agent style: ${error.detail || error.message || 'Unknown error'}`, 'danger');
        } finally {
            // Re-enable the textarea
            styleTextarea.disabled = false;
        }
    } catch (error) {
        // Show error
        Utils.showNotification(`Error autofilling agent style: ${error.detail || error.message || 'Unknown error'}`, 'danger');
    }
}
// Show clone agent modal
async function showCloneAgentModal() {
    try {
        // Show loading
        Utils.showLoading('clone-agents-container');
        
        // Get all agents the user has access to
        const companyId = document.getElementById('company-filter').value;
        let endpoint = '/agents';
        
        if (companyId) {
            endpoint += `?company_id=${companyId}`;
        }
        
        const agents = await API.get(endpoint);
        
        if (agents.length === 0) {
            Utils.hideLoading('clone-agents-container', '<p class="text-center">No agents found</p>');
            document.getElementById('confirm-clone').disabled = true;
            return;
        }
        
        let html = '<div class="row">';
        
        agents.forEach(agent => {
            html += `
                <div class="col-md-4 mb-3">
                    <div class="card agent-card h-100" data-id="${agent.agent_id}">
                        <div class="card-body">
                            <h5 class="card-title">${agent.agent_name}</h5>
                            <p class="card-text">${agent.description || 'No description'}</p>
                            <p class="mb-1"><small class="text-muted">Status: 
                                <span class="badge ${agent.on_status ? 'bg-success' : 'bg-danger'}">
                                    ${agent.on_status ? 'Active' : 'Inactive'}
                                </span>
                            </small></p>
                            <p class="mb-1"><small class="text-muted">Style: ${agent.agent_style || 'Default'}</small></p>
                            <p class="mb-1"><small class="text-muted">Company: ${agent.company_id ? 'Company Agent' : 'Personal Agent'}</small></p>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        
        Utils.hideLoading('clone-agents-container', html);
        
        // Add click event listeners to agent cards
        document.querySelectorAll('#clone-agents-container .agent-card').forEach(card => {
            card.addEventListener('click', function() {
                selectAgentToClone(this);
            });
        });
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('clone-agent-modal'));
        modal.show();
        
    } catch (error) {
        Utils.hideLoading('clone-agents-container', `<p class="text-center text-danger">Error loading agents: ${error.detail || error.message || 'Unknown error'}</p>`);
        Utils.showNotification(`Error loading agents: ${error.detail || error.message || 'Unknown error'}`, 'danger');
    }
}

// Select agent to clone
function selectAgentToClone(element) {
    // Clear previous selection
    document.querySelectorAll('#clone-agents-container .agent-card').forEach(card => {
        card.classList.remove('border-primary');
    });
    
    // Add border to selected agent
    element.classList.add('border-primary');
    
    // Enable confirm button
    document.getElementById('confirm-clone').disabled = false;
    
    // Store the selected agent ID as a data attribute on the confirm button
    document.getElementById('confirm-clone').setAttribute('data-agent-id', element.getAttribute('data-id'));
}

// Clone selected agent
async function cloneSelectedAgent() {
    const agentId = document.getElementById('confirm-clone').getAttribute('data-agent-id');
    
    if (!agentId) {
        Utils.showNotification('Please select an agent to clone', 'warning');
        return;
    }
    
    try {
        // Call the server endpoint to clone the agent
        const response = await API.post(`/agents/${agentId}/clone`);
        
        Utils.showNotification('Agent cloned successfully');
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('clone-agent-modal'));
        modal.hide();
        
        // Reload agents
        loadAgents();
        
        // Load the newly cloned agent details
        loadAgentDetails(response.agent_id);
        
    } catch (error) {
        Utils.showNotification(`Error cloning agent: ${error.detail || error.message || 'Unknown error'}`, 'danger');
    }
}
